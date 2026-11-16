# Demo-only build for the preprocessing pipeline.
# This is NOT the project build system; the repository Makefile is untouched.
#
# Usage:  make -f demo/Makefile          (run from the repo root)
#         make -f demo/Makefile clean

CC      ?= gcc
ROOT    := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
RUST_A  := $(ROOT)/rust/target/release/libbitgrain.a
OUT     := $(ROOT)/demo/preprocess_demo

CFLAGS  := -std=c11 -Wall -Wextra -O2 -I$(ROOT)/includes -I$(ROOT)/demo

# libwebp headers/libs, if a dev package is not installed system-wide.
WEBP_CFLAGS := $(shell pkg-config --cflags libwebp 2>/dev/null)
WEBP_LIBS   := $(shell pkg-config --libs libwebp 2>/dev/null)
ifeq ($(WEBP_LIBS),)
  WEBP_LIBS := -lwebp
endif

.PHONY: all clean
all: $(OUT)

$(RUST_A):
	@echo "Rust static library missing; run 'make bitgrain' at the repo root first." >&2
	@exit 1

$(OUT): $(ROOT)/c/metrics.c $(ROOT)/demo/main.c $(ROOT)/demo/preprocess_api.h $(RUST_A)
	$(CC) $(CFLAGS) $(WEBP_CFLAGS) \
	    $(ROOT)/demo/main.c $(ROOT)/c/metrics.c \
	    $(ROOT)/c/dct.c $(ROOT)/c/quant.c $(RUST_A) \
	    -o $(OUT) $(WEBP_LIBS) -lpthread -ldl -lm
	@echo "Built: $(OUT)"

clean:
	rm -f $(OUT)
