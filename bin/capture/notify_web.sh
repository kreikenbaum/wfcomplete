#! /bin/bash
### retrieves page from own http server to notify that sth was done
. config.py

. start_xvfb_if_necessary.sh
one_site.py gjem6zmaxxsztcoy.onion/
rm $SAVETO/gjem6zmaxxsztcoy.onion@*
