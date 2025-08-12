from django.db import models
from django.core.exceptions import ValidationError
from django.utils import timezone


class Label(models.Model):
    """Label model for storing classification labels."""
    name = models.CharField(max_length=255)
    percentage = models.FloatField()
    project = models.ForeignKey(
        'Project', related_name='labels', on_delete=models.CASCADE)

    class Meta:
        unique_together = ['name', 'project']

    def clean(self):
        if self.percentage < 0 or self.percentage > 100:
            raise ValidationError(
                "Percentage must be between 0 and 100")

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.percentage}%)"


class Project(models.Model):
    """Project model for storing project information."""

    title = models.CharField(max_length=255)
    description = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    organization = models.IntegerField(null=False)
    total_predictions_number = models.IntegerField(default=0)
    is_published = models.BooleanField(default=False)
    model_version = models.CharField(null=True, blank=True, max_length=255)
    is_draft = models.BooleanField(default=True)
    ml_type_name = models.CharField(null=True, blank=True, max_length=255)
    ls_project_id = models.IntegerField(unique=True, null=False)  # Id from the Label Studio project

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['created_at']),
            models.Index(fields=['updated_at']),
            models.Index(fields=['title']),
            models.Index(fields=['ml_type_name']),
        ]

    def __str__(self):
        return self.title

    def validate_label_percentages(self):
        total = sum(label.percentage for label in self.labels.all())
        return round(total, 2) == 100.0


class Prediction(models.Model):
    """Model for storing prediction metadata from S3 JSON files."""

    ls_project_id = models.BigIntegerField(
        help_text="Project ID from the prediction JSON")
    image_url = models.TextField(help_text="S3 URL of the predicted image")
    predicted_at = models.DateTimeField(
        help_text="Timestamp when prediction was made")
    json_file_name = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Name of the JSON file in S3"
    )
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-predicted_at']
        indexes = [
            models.Index(fields=['ls_project_id']),
            models.Index(fields=['predicted_at']),
            models.Index(fields=['json_file_name']),
        ]
        unique_together = ['ls_project_id', 'json_file_name']

    def __str__(self):
        return f"Prediction {self.id} - Project {self.ls_project_id} - {self.predicted_at}"


class PredictionResult(models.Model):
    """Model for storing individual prediction results."""

    prediction = models.ForeignKey(
        Prediction,
        related_name='results',
        on_delete=models.CASCADE
    )
    label = models.CharField(max_length=255, help_text="Class name/label")
    score = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        help_text="Prediction score with 2 decimal places"
    )
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['id']

    def __str__(self):
        return f"{self.label}: {self.score}%"
