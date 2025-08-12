from django.db import models
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _


User = get_user_model()


class Group(models.Model):
    name = models.CharField(_("Name"), max_length=255)
    user_ids = models.ManyToManyField(User, related_name="group_users", blank=True)
    projects = models.ManyToManyField(to='projects.Project', related_name="group_projects", blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name = "Group"
        verbose_name_plural = "Groups"
