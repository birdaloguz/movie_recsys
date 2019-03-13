from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class UserProfile(models.Model):
    user = models.OneToOneField(User)
    org_user_id = models.PositiveIntegerField(primary_key=True, default=1)
    description = models.CharField(max_length=100, default='')



