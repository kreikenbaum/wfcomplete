VAL=$(grep $1 ~/bin/tor-browser_en-US/Browser/TorBrowser/Data/Browser/profile.default/prefs.js | cut -d ' ' -f 2 | cut -d ')' -f 1)
echo ${VAL:-null}


