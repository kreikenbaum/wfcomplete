USER=$(shell cat local/moz_uid)
PASS=$(shell cat local/moz_pass)

sign: clean
	unset http_proxy; unset https_proxy; jpm -v sign --api-key $(USER) --api-secret $(PASS)

clean:
	rm install.rdf bootstrap.js || true
	mv *.xpi doc/oldxpi || true

lint:
	jshint js/* test/* index.js

jtest:
	jpm -b /usr/bin/firefox test

version: lint jtest
