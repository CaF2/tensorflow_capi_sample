#	Copyright (c) 2023 Florian Evaldsson <florian.evaldsson@telia.com>
#
#	Permission is hereby granted, free of charge, to any person obtaining a copy
#	of this software and associated documentation files (the "Software"), to deal
#	in the Software without restriction, including without limitation the rights
#	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#	copies of the Software, and to permit persons to whom the Software is
#	furnished to do so, subject to the following conditions:
#
#	The above copyright notice and this permission notice shall be included in all
#	copies or substantial portions of the Software.
#
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#	SOFTWARE.

-include user.mk

CC = gcc

NAME = main

ARGS =

TENSORFLOW_PKG = libtensorflow-cpu-linux-x86_64-2.11.0.tar.gz

ROOT_PATH = ..
CLIB_PATH = $(ROOT_PATH)/clib

###########

SRCS = $(wildcard *.c)

OBJS = $(SRCS:.c=.c.o)

PKG_CONF = glib-2.0

CFLAGS = $(if $(PKG_CONF),$(shell pkg-config --cflags $(PKG_CONF))) -g -I$(CLIB_PATH)/include 
CFLAGS += -MMD -MP

LDFLAGS = $(if $(PKG_CONF),$(shell pkg-config --libs $(PKG_CONF))) -lrt -lm -L$(CLIB_PATH)/lib -ltensorflow -ltensorflow_framework

CFLAGS += $(if $(NO_ASAN),,-fsanitize=address)
LDFLAGS += $(if $(NO_ASAN),,-fsanitize=address)

EXEC_PRE = LD_LIBRARY_PATH=$(CLIB_PATH)/lib

###########

all: $(NAME)

$(ROOT_PATH)/$(TENSORFLOW_PKG):
	cd $(ROOT_PATH); wget https://storage.googleapis.com/tensorflow/libtensorflow/$(TENSORFLOW_PKG)

$(CLIB_PATH): $(ROOT_PATH)/$(TENSORFLOW_PKG)
	mkdir -p $(CLIB_PATH)
	cd $(ROOT_PATH); tar -xvf $(TENSORFLOW_PKG) -C clib

lstm2:
	python model.py

$(NAME): $(OBJS) | lstm2
	$(CC) $^ -o $@ $(LDFLAGS)

%.c.o: %.c $(if NO_DL,,$(CLIB_PATH))
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
