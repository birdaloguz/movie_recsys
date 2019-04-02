from django.conf.urls import url
from quiz.views import *

urlpatterns = [
    url(r'^$', quiz.as_view(), name='quiz'),
    #url(r'^test/$', test, name='test'),
]