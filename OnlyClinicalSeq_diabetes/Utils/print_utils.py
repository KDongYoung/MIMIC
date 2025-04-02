import sys 

# 터미널과 로그 파일에 동시에 출력하는 클래스
class DualLogger:
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout  # 기존 터미널 출력 저장
        self.log = open(filename, "a")  # 로그 파일 (추가 모드)

    def write(self, message):
        self.terminal.write(message)  # 터미널 출력
        self.log.write(message)  # 파일 출력

    def flush(self):  # `sys.stdout.flush()` 오류 방지
        self.terminal.flush()
        self.log.flush()