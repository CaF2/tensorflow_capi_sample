-include user.mk

CC = gcc

NAME = main

ARGS =

TENSORFLOW_PKG = libtensorflow-cpu-linux-x86_64-2.11.0.tar.gz

###########

SRCS = $(wildcard *.c)

OBJS = $(SRCS:.c=.c.o)

PKG_CONF = glib-2.0

CFLAGS = $(if $(PKG_CONF),$(shell pkg-config --cflags $(PKG_CONF))) -g -Iclib/include 
CFLAGS += -MMD -MP

LDFLAGS = $(if $(PKG_CONF),$(shell pkg-config --libs $(PKG_CONF))) -lrt -lm -Lclib/lib -ltensorflow -ltensorflow_framework

CFLAGS += $(if $(NO_ASAN),,-fsanitize=address)
LDFLAGS += $(if $(NO_ASAN),,-fsanitize=address)

EXEC_PRE = LD_LIBRARY_PATH=clib/lib

###########

all: $(NAME)

$(TENSORFLOW_PKG):
	wget https://storage.googleapis.com/tensorflow/libtensorflow/$(TENSORFLOW_PKG)

clib: $(TENSORFLOW_PKG)
	mkdir -p clib
	tar -xvf $(TENSORFLOW_PKG) -C clib

lstm2:
	python model.py

$(NAME): $(OBJS) | lstm2
	$(CC) $^ -o $@ $(LDFLAGS)

%.c.o: %.c clib 
	$(CC) $< -c -o $@ $(CFLAGS)
	
run: all
	$(EXEC_PRE) ./$(NAME) $(ARGS)
	
gdb: all
	$(EXEC_PRE) gdb --args ./$(NAME) $(ARGS)

lldb: all
	$(EXEC_PRE) lldb -- ./$(NAME) $(ARGS)
	
valgrind: all
	$(EXEC_PRE) valgrind --leak-check=yes --leak-check=full --show-leak-kinds=all -v ./$(NAME) $(ARGS)

-include $(OBJS:.o=.d)
