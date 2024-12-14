#include "header.h"

void err_exit(const char *str) {
  perror(str);
  exit(EXIT_FAILURE);
}

void dirCat(char *filename, int id, const char *str) {
  char buf[3];
  char dir[100] = {0};

  sprintf(buf, "%d/", id);
  strcpy(dir, str);
  strcat(dir, buf);
  strcat(dir, filename);
  strcpy(filename, dir);
}

void writeDataToFile(const char *filename, int *data,
                     int data_size) {
  // 파일 디스크립터를 사용한 저수준 파일 생성 및 쓰기
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC,
                S_IRUSR | S_IWUSR);
  if (fd == -1) {
    perror("Error opening file for writing");
    exit(1);
  }

  ssize_t bytes_written =
      write(fd, data, data_size * sizeof(int));
  if (bytes_written == -1) {
    perror("Error writing data to file");
    close(fd);
    exit(1);
  }

  close(fd);
}

void createFilename(char *filename, const char *str1,
                    const char *str2, int id, int i) {
  // sm_id 추가
  int tmp_id = id;
  char str[10];
  int len = 0;
  int pos = 0;

  if (str1) strcat(filename, str1);

  if (id >= 0) {
    do {
      str[len++] = (tmp_id % 10) + '0';
      tmp_id /= 10;

    } while (tmp_id > 0);

    pos = strlen(filename);
    for (int j = len - 1; j >= 0; --j) {
      filename[pos++] = str[j];
    }
  }

  if (str2) strcat(filename, str2);

  // i 추가
  if (i >= 0) {
    tmp_id = i;
    len = 0;
    do {
      str[len++] = (tmp_id % 10) + '0';

      tmp_id /= 10;
    } while (tmp_id > 0);

    // i를 반대로 추가
    pos = strlen(filename);
    for (int j = len - 1; j >= 0; --j) {
      filename[pos++] = str[j];
    }
  }

  strcat(filename, ".txt");
}