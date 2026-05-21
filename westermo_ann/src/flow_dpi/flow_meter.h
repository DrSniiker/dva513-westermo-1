#ifndef FLOW_METER_H
#define FLOW_METER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/*
 * IPFIX-like 4-tuple flow key:
 * src_ip, dst_ip, dst_port, protocol.
 * IPv4 is stored as network-order 32-bit integers.
 */
typedef struct {
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t dst_port;
    uint8_t protocol;
    uint8_t _padding;
} flow_key_t;

/*
 * Memory-efficient flow record for edge metering.
 * Keep fields compact to sustain higher flow counts within 256 MB RAM.
 */
typedef struct flow_record_s {
    flow_key_t key;

    uint64_t first_seen_ms;
    uint64_t last_seen_ms;
    uint64_t bytes_total;
    uint32_t packets_total;

    uint16_t min_pkt_len;
    uint16_t max_pkt_len;
    uint32_t pkt_len_sum;

    uint32_t tcp_syn_count;
    uint32_t tcp_ack_count;
    uint32_t tcp_fin_count;
    uint32_t tcp_rst_count;
    uint32_t tcp_psh_count;
    uint32_t tcp_urg_count;

    uint32_t stnum_last;
    uint32_t stnum_transitions;
    uint32_t sqnum_last;
    uint32_t sqnum_out_of_order;

    uint32_t bit_string_hits;
    uint32_t ber_padding_anomaly_hits;

    uint32_t bucket_next; /* Chaining in bucket list */
    uint32_t lru_prev;    /* Doubly-linked active list */
    uint32_t lru_next;
    bool in_use;
} flow_record_t;

typedef struct {
    uint64_t observation_time_ms;
    uint64_t active_timeout_ms;
    uint64_t inactive_timeout_ms;
    uint32_t min_packet_size;
    uint32_t sampling_rate; /* 1 = keep all packets, 10 = keep ~10% */
    bool tcp_only;
    uint32_t random_seed;
} flow_meter_cfg_t;

typedef struct {
    uint8_t *data;
    uint32_t len;
    uint64_t ts_ms;
} packet_view_t;

typedef struct {
    flow_record_t *records;
    uint32_t *bucket_heads;
    uint32_t *free_stack;
    uint32_t lru_head;
    uint32_t lru_tail;
    uint32_t free_top;
    uint32_t max_flows;
    uint32_t bucket_count;
    flow_meter_cfg_t cfg;
} flow_meter_t;

typedef struct {
    flow_key_t key;
    uint64_t first_seen_ms;
    uint64_t last_seen_ms;
    uint32_t duration_ms;
    uint32_t packets_total;
    uint64_t bytes_total;
    float byte_rate_per_sec;
    float pkt_rate_per_sec;
    float avg_pkt_len;
    uint16_t min_pkt_len;
    uint16_t max_pkt_len;
    uint32_t tcp_syn_count;
    uint32_t tcp_ack_count;
    uint32_t tcp_fin_count;
    uint32_t tcp_rst_count;
    uint32_t tcp_psh_count;
    uint32_t tcp_urg_count;
    uint32_t stnum_transitions;
    uint32_t sqnum_out_of_order;
    uint32_t bit_string_hits;
    uint32_t ber_padding_anomaly_hits;
} flow_export_record_t;

typedef void (*flow_export_cb_t)(const flow_export_record_t *record, void *user_ctx);

/* Lifecycle */
bool flow_meter_init(flow_meter_t *meter, const flow_meter_cfg_t *cfg, uint32_t max_flows, uint32_t bucket_count);
void flow_meter_destroy(flow_meter_t *meter);

/* Packet processing + flow expiration/export */
void flow_meter_process_packet(flow_meter_t *meter, const packet_view_t *pkt, flow_export_cb_t on_export, void *user_ctx);
void flow_meter_expire(flow_meter_t *meter, uint64_t now_ms, flow_export_cb_t on_export, void *user_ctx);
void flow_meter_flush_all(flow_meter_t *meter, flow_export_cb_t on_export, void *user_ctx);

/* Utility for debugging / CSV export of records */
void flow_export_record_to_csv(FILE *out, const flow_export_record_t *record);

#endif /* FLOW_METER_H */
