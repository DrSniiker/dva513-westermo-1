#include "flow_meter.h"

#include <arpa/inet.h>
#include <stdlib.h>
#include <string.h>

#define FLOW_INVALID_INDEX UINT32_MAX
#define ETH_TYPE_IPV4 0x0800u
#define IP_PROTO_TCP 6u
#define IP_PROTO_UDP 17u

typedef struct {
    uint8_t dst_mac[6];
    uint8_t src_mac[6];
    uint16_t eth_type;
} __attribute__((packed)) eth_hdr_t;

typedef struct {
    uint8_t version_ihl;
    uint8_t dscp_ecn;
    uint16_t total_len;
    uint16_t identification;
    uint16_t flags_frag;
    uint8_t ttl;
    uint8_t protocol;
    uint16_t checksum;
    uint32_t src_ip;
    uint32_t dst_ip;
} __attribute__((packed)) ipv4_hdr_t;

typedef struct {
    uint16_t src_port;
    uint16_t dst_port;
    uint32_t seq;
    uint32_t ack;
    uint8_t data_offset_reserved;
    uint8_t flags;
    uint16_t window;
    uint16_t checksum;
    uint16_t urgent_ptr;
} __attribute__((packed)) tcp_hdr_t;

typedef struct {
    uint16_t src_port;
    uint16_t dst_port;
    uint16_t length;
    uint16_t checksum;
} __attribute__((packed)) udp_hdr_t;

static uint32_t flow_hash(const flow_key_t *key, uint32_t bucket_count) {
    uint32_t h = 2166136261u;
    h ^= key->src_ip;
    h *= 16777619u;
    h ^= key->dst_ip;
    h *= 16777619u;
    h ^= key->dst_port;
    h *= 16777619u;
    h ^= key->protocol;
    h *= 16777619u;
    return h % bucket_count;
}

static bool flow_key_equal(const flow_key_t *a, const flow_key_t *b) {
    return (
        a->src_ip == b->src_ip &&
        a->dst_ip == b->dst_ip &&
        a->dst_port == b->dst_port &&
        a->protocol == b->protocol
    );
}

static uint32_t meter_rand(flow_meter_t *meter) {
    uint32_t x = meter->cfg.random_seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    meter->cfg.random_seed = x;
    return x;
}

static bool packet_selected(flow_meter_t *meter, const packet_view_t *pkt) {
    if (meter->cfg.min_packet_size > 0 && pkt->len < meter->cfg.min_packet_size) {
        return false;
    }
    if (meter->cfg.sampling_rate <= 1) {
        return true;
    }
    return (meter_rand(meter) % meter->cfg.sampling_rate) == 0;
}

static bool parse_packet(
    const packet_view_t *pkt,
    flow_key_t *key_out,
    uint16_t *pkt_len_out,
    uint8_t *tcp_flags_out,
    const uint8_t **payload_out,
    uint16_t *payload_len_out
) {
    if (pkt->len < sizeof(eth_hdr_t)) {
        return false;
    }

    const eth_hdr_t *eth = (const eth_hdr_t *) pkt->data;
    if (ntohs(eth->eth_type) != ETH_TYPE_IPV4) {
        return false;
    }

    const uint8_t *ip_ptr = pkt->data + sizeof(eth_hdr_t);
    uint32_t ip_available = pkt->len - (uint32_t) sizeof(eth_hdr_t);
    if (ip_available < sizeof(ipv4_hdr_t)) {
        return false;
    }

    const ipv4_hdr_t *ip = (const ipv4_hdr_t *) ip_ptr;
    uint8_t ihl_words = ip->version_ihl & 0x0fu;
    uint16_t ip_hlen = (uint16_t) ihl_words * 4u;
    if (ip_hlen < 20u || ip_available < ip_hlen) {
        return false;
    }

    uint8_t protocol = ip->protocol;
    uint16_t dst_port = 0;
    uint8_t tcp_flags = 0;
    const uint8_t *l4_ptr = ip_ptr + ip_hlen;
    uint32_t l4_available = ip_available - ip_hlen;
    const uint8_t *payload = NULL;
    uint16_t payload_len = 0;

    if (protocol == IP_PROTO_TCP) {
        if (l4_available < sizeof(tcp_hdr_t)) {
            return false;
        }
        const tcp_hdr_t *tcp = (const tcp_hdr_t *) l4_ptr;
        uint8_t doff_words = (tcp->data_offset_reserved >> 4) & 0x0fu;
        uint16_t tcp_hlen = (uint16_t) doff_words * 4u;
        if (tcp_hlen < 20u || l4_available < tcp_hlen) {
            return false;
        }
        dst_port = ntohs(tcp->dst_port);
        tcp_flags = tcp->flags;
        payload = l4_ptr + tcp_hlen;
        payload_len = (uint16_t) (l4_available - tcp_hlen);
    } else if (protocol == IP_PROTO_UDP) {
        if (l4_available < sizeof(udp_hdr_t)) {
            return false;
        }
        const udp_hdr_t *udp = (const udp_hdr_t *) l4_ptr;
        dst_port = ntohs(udp->dst_port);
        payload = l4_ptr + sizeof(udp_hdr_t);
        payload_len = (uint16_t) (l4_available - sizeof(udp_hdr_t));
    } else {
        return false;
    }

    key_out->src_ip = ip->src_ip;
    key_out->dst_ip = ip->dst_ip;
    key_out->dst_port = dst_port;
    key_out->protocol = protocol;
    key_out->_padding = 0;
    *pkt_len_out = (uint16_t) pkt->len;
    *tcp_flags_out = tcp_flags;
    *payload_out = payload;
    *payload_len_out = payload_len;
    return true;
}

static void lru_remove(flow_meter_t *meter, uint32_t idx) {
    flow_record_t *r = &meter->records[idx];
    if (r->lru_prev != FLOW_INVALID_INDEX) {
        meter->records[r->lru_prev].lru_next = r->lru_next;
    } else {
        meter->lru_head = r->lru_next;
    }
    if (r->lru_next != FLOW_INVALID_INDEX) {
        meter->records[r->lru_next].lru_prev = r->lru_prev;
    } else {
        meter->lru_tail = r->lru_prev;
    }
    r->lru_prev = FLOW_INVALID_INDEX;
    r->lru_next = FLOW_INVALID_INDEX;
}

static void lru_touch_tail(flow_meter_t *meter, uint32_t idx) {
    flow_record_t *r = &meter->records[idx];
    if (meter->lru_tail == idx) {
        return;
    }
    if (r->lru_prev != FLOW_INVALID_INDEX || r->lru_next != FLOW_INVALID_INDEX || meter->lru_head == idx) {
        lru_remove(meter, idx);
    }

    r->lru_prev = meter->lru_tail;
    r->lru_next = FLOW_INVALID_INDEX;
    if (meter->lru_tail != FLOW_INVALID_INDEX) {
        meter->records[meter->lru_tail].lru_next = idx;
    } else {
        meter->lru_head = idx;
    }
    meter->lru_tail = idx;
}

static void export_record(const flow_record_t *r, flow_export_cb_t on_export, void *user_ctx) {
    if (on_export == NULL) {
        return;
    }
    flow_export_record_t out;
    memset(&out, 0, sizeof(out));
    out.key = r->key;
    out.first_seen_ms = r->first_seen_ms;
    out.last_seen_ms = r->last_seen_ms;
    out.duration_ms = (uint32_t) (r->last_seen_ms - r->first_seen_ms);
    out.packets_total = r->packets_total;
    out.bytes_total = r->bytes_total;
    out.min_pkt_len = r->min_pkt_len;
    out.max_pkt_len = r->max_pkt_len;
    out.avg_pkt_len = (r->packets_total > 0) ? ((float) r->pkt_len_sum / (float) r->packets_total) : 0.0f;

    if (out.duration_ms == 0u) {
        out.byte_rate_per_sec = (float) out.bytes_total;
        out.pkt_rate_per_sec = (float) out.packets_total;
    } else {
        float dur_s = (float) out.duration_ms / 1000.0f;
        out.byte_rate_per_sec = (float) out.bytes_total / dur_s;
        out.pkt_rate_per_sec = (float) out.packets_total / dur_s;
    }

    out.tcp_syn_count = r->tcp_syn_count;
    out.tcp_ack_count = r->tcp_ack_count;
    out.tcp_fin_count = r->tcp_fin_count;
    out.tcp_rst_count = r->tcp_rst_count;
    out.tcp_psh_count = r->tcp_psh_count;
    out.tcp_urg_count = r->tcp_urg_count;
    out.stnum_transitions = r->stnum_transitions;
    out.sqnum_out_of_order = r->sqnum_out_of_order;
    out.bit_string_hits = r->bit_string_hits;
    out.ber_padding_anomaly_hits = r->ber_padding_anomaly_hits;
    on_export(&out, user_ctx);
}

static void remove_from_bucket(flow_meter_t *meter, uint32_t idx) {
    flow_record_t *r = &meter->records[idx];
    uint32_t bucket = flow_hash(&r->key, meter->bucket_count);
    uint32_t cur = meter->bucket_heads[bucket];
    uint32_t prev = FLOW_INVALID_INDEX;

    while (cur != FLOW_INVALID_INDEX) {
        if (cur == idx) {
            if (prev == FLOW_INVALID_INDEX) {
                meter->bucket_heads[bucket] = meter->records[cur].bucket_next;
            } else {
                meter->records[prev].bucket_next = meter->records[cur].bucket_next;
            }
            break;
        }
        prev = cur;
        cur = meter->records[cur].bucket_next;
    }
}

static void release_flow(flow_meter_t *meter, uint32_t idx, flow_export_cb_t on_export, void *user_ctx) {
    flow_record_t *r = &meter->records[idx];
    export_record(r, on_export, user_ctx);
    remove_from_bucket(meter, idx);
    lru_remove(meter, idx);
    memset(r, 0, sizeof(*r));
    r->bucket_next = FLOW_INVALID_INDEX;
    r->lru_prev = FLOW_INVALID_INDEX;
    r->lru_next = FLOW_INVALID_INDEX;
    r->in_use = false;
    meter->free_stack[meter->free_top++] = idx;
}

static uint32_t alloc_flow(flow_meter_t *meter) {
    if (meter->free_top == 0u) {
        return FLOW_INVALID_INDEX;
    }
    uint32_t idx = meter->free_stack[--meter->free_top];
    flow_record_t *r = &meter->records[idx];
    memset(r, 0, sizeof(*r));
    r->bucket_next = FLOW_INVALID_INDEX;
    r->lru_prev = FLOW_INVALID_INDEX;
    r->lru_next = FLOW_INVALID_INDEX;
    r->in_use = true;
    return idx;
}

static bool payload_has_token(const uint8_t *payload, uint16_t payload_len, const char *token) {
    size_t token_len = strlen(token);
    if (token_len == 0 || payload_len < token_len) {
        return false;
    }
    for (uint16_t i = 0; i <= payload_len - token_len; i++) {
        if (memcmp(payload + i, token, token_len) == 0) {
            return true;
        }
    }
    return false;
}

static bool payload_extract_uint(const uint8_t *payload, uint16_t payload_len, const char *token, uint32_t *out_value) {
    size_t token_len = strlen(token);
    if (token_len == 0 || payload_len <= token_len) {
        return false;
    }
    for (uint16_t i = 0; i <= payload_len - token_len; i++) {
        if (memcmp(payload + i, token, token_len) != 0) {
            continue;
        }
        uint32_t v = 0u;
        bool has_digit = false;
        for (uint16_t j = (uint16_t) (i + token_len); j < payload_len; j++) {
            uint8_t c = payload[j];
            if (c >= '0' && c <= '9') {
                has_digit = true;
                v = (v * 10u) + (uint32_t) (c - '0');
            } else {
                break;
            }
        }
        if (has_digit) {
            *out_value = v;
            return true;
        }
    }
    return false;
}

static bool has_ber_padding_anomaly(const uint8_t *payload, uint16_t payload_len) {
    if (payload_len < 2u) {
        return false;
    }
    uint8_t pad = payload[payload_len - 1u];
    if (pad == 0u || pad > 7u) {
        return false;
    }
    uint8_t marker = payload[payload_len - 2u];
    return marker == 0x03u;
}

bool flow_meter_init(flow_meter_t *meter, const flow_meter_cfg_t *cfg, uint32_t max_flows, uint32_t bucket_count) {
    if (meter == NULL || cfg == NULL || max_flows == 0u || bucket_count == 0u) {
        return false;
    }

    memset(meter, 0, sizeof(*meter));
    meter->records = (flow_record_t *) calloc(max_flows, sizeof(flow_record_t));
    meter->bucket_heads = (uint32_t *) malloc(bucket_count * sizeof(uint32_t));
    meter->free_stack = (uint32_t *) malloc(max_flows * sizeof(uint32_t));
    if (meter->records == NULL || meter->bucket_heads == NULL || meter->free_stack == NULL) {
        flow_meter_destroy(meter);
        return false;
    }

    meter->max_flows = max_flows;
    meter->bucket_count = bucket_count;
    meter->cfg = *cfg;
    meter->lru_head = FLOW_INVALID_INDEX;
    meter->lru_tail = FLOW_INVALID_INDEX;
    meter->free_top = max_flows;

    for (uint32_t i = 0; i < bucket_count; i++) {
        meter->bucket_heads[i] = FLOW_INVALID_INDEX;
    }
    for (uint32_t i = 0; i < max_flows; i++) {
        meter->records[i].bucket_next = FLOW_INVALID_INDEX;
        meter->records[i].lru_prev = FLOW_INVALID_INDEX;
        meter->records[i].lru_next = FLOW_INVALID_INDEX;
        meter->free_stack[i] = max_flows - 1u - i;
    }
    return true;
}

void flow_meter_destroy(flow_meter_t *meter) {
    if (meter == NULL) {
        return;
    }
    free(meter->records);
    free(meter->bucket_heads);
    free(meter->free_stack);
    memset(meter, 0, sizeof(*meter));
}

void flow_meter_expire(flow_meter_t *meter, uint64_t now_ms, flow_export_cb_t on_export, void *user_ctx) {
    if (meter == NULL) {
        return;
    }

    uint32_t cur = meter->lru_head;
    while (cur != FLOW_INVALID_INDEX) {
        flow_record_t *r = &meter->records[cur];
        uint32_t next = r->lru_next;
        bool inactive_expired = (now_ms - r->last_seen_ms) >= meter->cfg.inactive_timeout_ms;
        bool active_expired = (now_ms - r->first_seen_ms) >= meter->cfg.active_timeout_ms;
        if (inactive_expired || active_expired) {
            release_flow(meter, cur, on_export, user_ctx);
        } else {
            break; /* LRU head is fresh enough; newer entries are fresher. */
        }
        cur = next;
    }
}

void flow_meter_process_packet(flow_meter_t *meter, const packet_view_t *pkt, flow_export_cb_t on_export, void *user_ctx) {
    if (meter == NULL || pkt == NULL || pkt->data == NULL || pkt->len == 0u) {
        return;
    }
    meter->cfg.observation_time_ms = pkt->ts_ms;
    flow_meter_expire(meter, pkt->ts_ms, on_export, user_ctx);

    if (!packet_selected(meter, pkt)) {
        return;
    }

    flow_key_t key;
    uint16_t pkt_len = 0;
    uint8_t tcp_flags = 0;
    const uint8_t *payload = NULL;
    uint16_t payload_len = 0;
    if (!parse_packet(pkt, &key, &pkt_len, &tcp_flags, &payload, &payload_len)) {
        return;
    }
    if (meter->cfg.tcp_only && key.protocol != IP_PROTO_TCP) {
        return;
    }

    uint32_t bucket = flow_hash(&key, meter->bucket_count);
    uint32_t idx = meter->bucket_heads[bucket];
    while (idx != FLOW_INVALID_INDEX && !flow_key_equal(&meter->records[idx].key, &key)) {
        idx = meter->records[idx].bucket_next;
    }

    if (idx == FLOW_INVALID_INDEX) {
        idx = alloc_flow(meter);
        if (idx == FLOW_INVALID_INDEX) {
            if (meter->lru_head != FLOW_INVALID_INDEX) {
                release_flow(meter, meter->lru_head, on_export, user_ctx);
                idx = alloc_flow(meter);
            }
            if (idx == FLOW_INVALID_INDEX) {
                return;
            }
        }
        flow_record_t *r = &meter->records[idx];
        r->key = key;
        r->first_seen_ms = pkt->ts_ms;
        r->last_seen_ms = pkt->ts_ms;
        r->min_pkt_len = pkt_len;
        r->max_pkt_len = pkt_len;
        r->stnum_last = UINT32_MAX;
        r->sqnum_last = UINT32_MAX;

        r->bucket_next = meter->bucket_heads[bucket];
        meter->bucket_heads[bucket] = idx;
        lru_touch_tail(meter, idx);
    }

    flow_record_t *r = &meter->records[idx];
    r->last_seen_ms = pkt->ts_ms;
    r->packets_total += 1u;
    r->bytes_total += pkt_len;
    r->pkt_len_sum += pkt_len;
    if (pkt_len < r->min_pkt_len) {
        r->min_pkt_len = pkt_len;
    }
    if (pkt_len > r->max_pkt_len) {
        r->max_pkt_len = pkt_len;
    }

    if (key.protocol == IP_PROTO_TCP) {
        if ((tcp_flags & 0x02u) != 0u) r->tcp_syn_count += 1u;
        if ((tcp_flags & 0x10u) != 0u) r->tcp_ack_count += 1u;
        if ((tcp_flags & 0x01u) != 0u) r->tcp_fin_count += 1u;
        if ((tcp_flags & 0x04u) != 0u) r->tcp_rst_count += 1u;
        if ((tcp_flags & 0x08u) != 0u) r->tcp_psh_count += 1u;
        if ((tcp_flags & 0x20u) != 0u) r->tcp_urg_count += 1u;
    }

    /*
     * Simple payload-based extraction placeholders for IEC metadata.
     * In production, swap with strict ASN.1/GOOSE parsing.
     */
    if (payload != NULL && payload_len > 0u) {
        uint32_t stnum = 0u;
        if (payload_extract_uint(payload, payload_len, "stNum=", &stnum)) {
            if (r->stnum_last != UINT32_MAX && stnum != r->stnum_last) {
                r->stnum_transitions += 1u;
            }
            r->stnum_last = stnum;
        }
        uint32_t sqnum = 0u;
        if (payload_extract_uint(payload, payload_len, "sqNum=", &sqnum)) {
            if (r->sqnum_last != UINT32_MAX && sqnum < r->sqnum_last) {
                r->sqnum_out_of_order += 1u;
            }
            r->sqnum_last = sqnum;
        }
        if (payload_has_token(payload, payload_len, "bit_string")) {
            r->bit_string_hits += 1u;
        }
        if (has_ber_padding_anomaly(payload, payload_len)) {
            r->ber_padding_anomaly_hits += 1u;
        }
    }

    lru_touch_tail(meter, idx);
}

void flow_meter_flush_all(flow_meter_t *meter, flow_export_cb_t on_export, void *user_ctx) {
    if (meter == NULL) {
        return;
    }
    while (meter->lru_head != FLOW_INVALID_INDEX) {
        release_flow(meter, meter->lru_head, on_export, user_ctx);
    }
}

void flow_export_record_to_csv(FILE *out, const flow_export_record_t *record) {
    if (out == NULL || record == NULL) {
        return;
    }
    fprintf(
        out,
        "%u,%u,%u,%u,%llu,%llu,%u,%llu,%.3f,%.3f,%.3f,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
        ntohl(record->key.src_ip),
        ntohl(record->key.dst_ip),
        record->key.dst_port,
        record->key.protocol,
        (unsigned long long) record->first_seen_ms,
        (unsigned long long) record->last_seen_ms,
        record->packets_total,
        (unsigned long long) record->bytes_total,
        record->byte_rate_per_sec,
        record->pkt_rate_per_sec,
        record->avg_pkt_len,
        record->min_pkt_len,
        record->max_pkt_len,
        record->tcp_syn_count,
        record->tcp_ack_count,
        record->tcp_fin_count,
        record->tcp_rst_count,
        record->tcp_psh_count,
        record->tcp_urg_count,
        record->stnum_transitions,
        record->sqnum_out_of_order,
        record->bit_string_hits,
        record->ber_padding_anomaly_hits
    );
}
