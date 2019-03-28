# -*- coding: utf-8 -*-
# Generated by Django 1.11.6 on 2019-03-13 21:46
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('movies', '0002_auto_20190313_2244'),
    ]

    operations = [
        migrations.AlterField(
            model_name='link',
            name='movie_id',
            field=models.PositiveIntegerField(),
        ),
        migrations.AlterField(
            model_name='rating',
            name='movie_id',
            field=models.PositiveIntegerField(),
        ),
        migrations.AlterField(
            model_name='rating',
            name='user_id',
            field=models.PositiveIntegerField(),
        ),
        migrations.AlterField(
            model_name='tag',
            name='movie_id',
            field=models.PositiveIntegerField(),
        ),
        migrations.AlterField(
            model_name='tag',
            name='user_id',
            field=models.PositiveIntegerField(),
        ),
    ]