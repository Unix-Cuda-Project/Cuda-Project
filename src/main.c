#include "header.h"

int main(int argc, char *argv[]) {
  pid_t pid;

  char *av[10] = {"./a.out", NULL, NULL, NULL};

  key_t key = ftok(".", 65);
  int msgid;

  msgid = msgget(key, 0666 | IPC_CREAT);

  for (int i = 0; i < 8; ++i) {
    av[1] = malloc(sizeof(char) * 10);
    av[2] = malloc(sizeof(char) * 10);
    sprintf(av[1], "%d", i);
    strcpy(av[2], argv[1]);

    pid = fork();
    if (pid == 0) {
      execv("./a.out", av);
      exit(1);
    } else if (pid > 0) {
    }
  }

  for (int i = 0; i < 8; ++i) wait(NULL);

  msgctl(msgid, IPC_RMID, NULL);
  return 0;
}
