# Qwen3TTS-Streaming — Development Makefile
#
# Usage:
#   make proto          Regenerate protobuf Python files from proto/tts.proto
#   make proto-sync     Copy generated proto files to engine/ and client/
#   make proto-check    Verify that consumer copies match the source

.PHONY: proto proto-sync proto-check

PROTO_DIR := proto
PROTO_FILE := $(PROTO_DIR)/tts.proto

ENGINE_PROTO_DIR := engine/gateway
CLIENT_PROTO_DIR := client/src/qwen3tts/_proto

proto:
	python -m grpc_tools.protoc \
		-I $(PROTO_DIR) \
		--python_out=$(PROTO_DIR) \
		--grpc_python_out=$(PROTO_DIR) \
		$(PROTO_FILE)

proto-sync:
	cp $(PROTO_DIR)/tts_pb2.py $(ENGINE_PROTO_DIR)/tts_pb2.py
	cp $(PROTO_DIR)/tts_pb2_grpc.py $(ENGINE_PROTO_DIR)/tts_pb2_grpc.py
	cp $(PROTO_DIR)/tts_pb2.py $(CLIENT_PROTO_DIR)/tts_pb2.py
	cp $(PROTO_DIR)/tts_pb2_grpc.py $(CLIENT_PROTO_DIR)/tts_pb2_grpc.py

proto-check:
	@diff $(PROTO_DIR)/tts_pb2.py $(ENGINE_PROTO_DIR)/tts_pb2.py && \
	 diff $(PROTO_DIR)/tts_pb2_grpc.py $(ENGINE_PROTO_DIR)/tts_pb2_grpc.py && \
	 diff $(PROTO_DIR)/tts_pb2.py $(CLIENT_PROTO_DIR)/tts_pb2.py && \
	 diff $(PROTO_DIR)/tts_pb2_grpc.py $(CLIENT_PROTO_DIR)/tts_pb2_grpc.py && \
	 echo "All proto copies are in sync." || \
	 (echo "ERROR: Proto copies are out of sync. Run 'make proto-sync'." && exit 1)
