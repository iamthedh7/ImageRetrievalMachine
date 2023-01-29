from django.db import models

# Create your models here.

class Search(models.Model):
    img = models.ImageField(upload_to = "images/upload", blank=True)
    topk = models.IntegerField(default=10)