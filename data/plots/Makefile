SRC=$(shell ls | grep -vE '(png|all|Makefile|plot|p1|log)')

plot:
	gnuplot plot.gnuplot

all: clean
	cat ${SRC} > all

clean:
	rm all || true

