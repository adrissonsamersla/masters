"""Generates protocol messages."""

from grpc_tools import protoc

protoc.main(
    (
        '',
        '-I./proto',
        '--python_out=./codegen',
        './proto/smallsize.proto',
    )
)
