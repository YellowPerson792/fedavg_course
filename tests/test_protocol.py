import socket

from fedavg.protocol import recv_message, send_message


def test_send_recv_message_round_trip():
    left, right = socket.socketpair()
    try:
        payload = b"abc123"
        sent = send_message(left, "REGISTER", {"client_id": "client0"}, payload)
        message = recv_message(right)
        assert sent > len(payload)
        assert message.msg_type == "REGISTER"
        assert message.metadata["client_id"] == "client0"
        assert message.payload == payload
        assert message.raw_bytes == sent
    finally:
        left.close()
        right.close()
