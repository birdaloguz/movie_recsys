from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Post(models.Model):
    post = models.CharField(max_length=500)
    user = models.ForeignKey(User)
    date = models.DateTimeField(auto_now=True)