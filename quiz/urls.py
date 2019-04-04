from django.conf.urls import url
from quiz.views import *

urlpatterns = [
    url(r'^mf/$', quiz_mf.as_view(), name='quiz_mf'),
    url(r'^knn/$', quiz_knn.as_view(), name='quiz_knn'),
    #url(r'^test/$', test, name='test'),
]