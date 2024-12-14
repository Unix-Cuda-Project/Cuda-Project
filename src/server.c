#include "header.h"

void child_run(int server_id) {
  int server_fd, client_fd, len;
  struct sockaddr_in address;
  socklen_t addrlen = sizeof(address);
  int buffer[BUFFER_SIZE];

  // 2. 서버 소켓 생성
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0)
    err_exit("Socket failed");

  // 3. 소켓 옵션 설정 (포트 재사용)
  int opt = 1;
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt,
                 sizeof(opt)) < 0)
    err_exit("Setsockopt failed");

  // 4. 서버 주소 설정
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(PORT + server_id);

  // 5. 소켓 바인딩
  if (bind(server_fd, (struct sockaddr *)&address,
           sizeof(address)) < 0)
    err_exit("Bind failed");

  // 6. 서버 리슨
  if (listen(server_fd, 8) < 0) err_exit("Listen failed");

  for (int i = 0; i < 8; ++i) {
    char filename[100] = {0};

    client_fd = accept(
        server_fd, (struct sockaddr *)&address, &addrlen);
    len = recv(client_fd, buffer, sizeof(buffer), 0);

    createFilename(filename, "server", "_", server_id,
                   buffer[0] / 2048);
    dirCat(filename, server_id, "server");
    writeDataToFile(filename, buffer, len / 4);
    close(client_fd);
  }
}

int main() {
  for (int i = 0; i < 2; ++i) {
    switch (fork()) {
      case -1:
        perror("fork");
        exit(1);
        break;
      case 0:
        child_run(i);
        exit(0);
    }
  }

  for (int i = 0; i < 2; ++i) wait(NULL);
  return 0;
}