### gets configuration of addon: factor
DEFAULT=50
get_pref.sh 'extensions.@wf-cover.factor' | sed "s/null/$DEFAULT/g"

