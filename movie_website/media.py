"""
本模块建立一个Movie类，用于封装movie的属性。
"""

class Movie(object):
    # This class provides a way to store movie related information

    def __init__(self, title, director, release_date, image_url, video_url):
        # initialize instance of class Movie
        self.title = title
        self.director = director
        self.release_date = release_date
        self.poster_image_url = image_url
        self.trailer_youtube_url = video_url