# Generated by Django 4.1.3 on 2023-02-01 15:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('homepage', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='search',
            name='use_full',
            field=models.CharField(default='off', max_length=10),
        ),
    ]
