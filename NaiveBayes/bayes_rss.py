import feedparser

# url = "http://newyork.craigslist.org/stp/index.rss"
url = "http://www.nasa.gov/rss/dyn/image_of_the_day.rss"
ny = feedparser.parse(url)
print(ny['entries'])
print(len(ny['entries']))
