sizes = []
urls = []
combined = []

for line in open('./urls'):
    (size, url) = line.split()
    sizes.append(int(size))
    urls.append('http://' + url)
    combined.append({'size': int(size), 'url': 'http://' + url})
