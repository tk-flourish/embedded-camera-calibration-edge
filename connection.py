
import socket

class Request:
    command: str
    data: bytes

    def __init__(self, command: str, argument: bytes) -> None:
        self.command = command
        self.data = argument

class ClientConnection:
    _socket: socket.socket

    def __init__(self, _socket: socket.socket) -> None:
        self._socket = _socket

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._socket.close()

    def read_raw(self) -> bytes | None:
        length_raw = self._socket.recv(4)
        if length_raw == b'': return None
        length = int.from_bytes(length_raw, 'big')
        return self._socket.recv(length)
    
    def read(self) -> Request | None:
        raw_data = self.read_raw()
        if raw_data == None: return None
        command_len = int.from_bytes(raw_data[0:4], 'big')
        command = raw_data[4:(command_len + 4)].decode()
        return Request(command, raw_data[(command_len + 4):])

    def send_raw(self, data: bytes) -> str:
        self._socket.sendall(len(data).to_bytes(4, 'big'))
        self._socket.sendall(data)

    def send(self, status: str, data: bytes = bytes([])):
        send_data = bytearray([])
        status_bytes = bytes(status, 'utf-8')
        send_data.extend(len(status_bytes).to_bytes(4, 'big'))
        send_data.extend(status_bytes)
        send_data.extend(data)
        self.send_raw(bytes(send_data))

class ServerStream:
    _socket: socket.socket

    def __init__(self, port: int) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind(("0.0.0.0", port))
        self._socket.listen()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._socket.close()
    
    def accept(self) -> tuple[ClientConnection, tuple[str, int]]:
        conn, addr = self._socket.accept()
        return (ClientConnection(conn), addr)