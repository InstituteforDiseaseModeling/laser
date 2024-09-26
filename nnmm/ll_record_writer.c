#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 1024 * 1024 // Size of buffer in bytes (1 MB)

static uint32_t* buffer = NULL;
static size_t buffer_count = 0;
static size_t max_buffer_count;
static FILE* file = NULL;

// Initialize the buffer and open the file
void init_writer(const char* filename) {
    file = fopen(filename, "wb");  // Open file in write-binary mode
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    
    // Allocate memory for the buffer (BUFFER_SIZE bytes)
    max_buffer_count = BUFFER_SIZE / (4 * sizeof(uint32_t));  // 4 integers per record
    buffer = (uint32_t*) malloc(BUFFER_SIZE);
    buffer_count = 0;  // Start with an empty buffer
}

// Flush the buffer to the file
void flush_buffer() {
    if (buffer_count > 0 && file != NULL) {
        fwrite(buffer, sizeof(uint32_t), buffer_count * 4, file);  // Write all buffered records
        buffer_count = 0;  // Reset the buffer
    }
}

// Write a record to the buffer
void write_record(uint32_t agent_id, uint32_t age_at_infection, uint32_t time_at_infection, uint32_t node_at_infection) {
    if (buffer_count >= max_buffer_count) {
        flush_buffer();  // If the buffer is full, write to disk
    }
    
    // Store the record in the buffer
    size_t index = buffer_count * 4;  // 4 integers per record
    buffer[index] = agent_id;
    buffer[index + 1] = age_at_infection;
    buffer[index + 2] = time_at_infection;
    buffer[index + 3] = node_at_infection;
    buffer_count++;
}

// Write multiple records to the buffer
void write_records_batch(uint32_t *agent_ids, uint32_t *ages_at_infection, uint32_t *times_at_infection, uint32_t *nodes_at_infection, size_t num_records) {
    for (size_t i = 0; i < num_records; i++) {
        // If the buffer is full, flush it to disk
        if (buffer_count >= max_buffer_count) {
            flush_buffer();
        }

        // Store each record in the buffer
        size_t index = buffer_count * 4;  // 4 integers per record
        buffer[index] = agent_ids[i];
        buffer[index + 1] = ages_at_infection[i];
        buffer[index + 2] = times_at_infection[i];
        buffer[index + 3] = nodes_at_infection[i];
        buffer_count++;
    }
}

// Close the writer and flush remaining data
void close_writer() {
    if (file != NULL) {
        flush_buffer();  // Write any remaining buffered data
        fclose(file);
        file = NULL;
    }
    
    if (buffer != NULL) {
        free(buffer);  // Free the allocated buffer memory
        buffer = NULL;
    }
}

