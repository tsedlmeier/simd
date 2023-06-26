CFLAGS= -std=c99 -Wextra -Wall -mavx512f -mavx
LDFLAGS= 

.PHONY: all

all: debug matmul

run: 
	./matmul

debug: matmul.c
	gcc -std=c99 -g  $(CFLAGS) $< -o $@

matmul: matmul.c
	gcc $(CFLAGS) $< -o $@

clean:
	rm -rf debug matmul
