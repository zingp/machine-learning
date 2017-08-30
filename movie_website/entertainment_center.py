"""
主程序，运行该文件可访问网站首页。
"""
import media
import settings
import fresh_tomatoes

def load_movie_info():
    """从配置文件中读取各部电影的信息并返回"""
    for movie_info in settings.movie_info_list:
        yield movie_info

def gene_movie_obj(info_list):
    """根据列表形式的电影信息，返回Movie实例"""
    for m in info_list:
        movie_obj = media.Movie(m[0], m[1], m[2], m[3], m[4])
        yield movie_obj


movies_info = load_movie_info()
movies = [i for i in gene_movie_obj(movies_info)]
fresh_tomatoes.open_movies_page(movies)
