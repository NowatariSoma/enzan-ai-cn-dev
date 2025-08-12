from rest_framework import serializers
from .models import Project, Label, Prediction, PredictionResult


class LabelSerializer(serializers.ModelSerializer):
    """Serializer for the Label model."""

    class Meta:
        model = Label
        fields = ['name', 'percentage']
        read_only_fields = ['name', 'percentage']


class ProjectSerializer(serializers.ModelSerializer):
    """Serializer for the Project model."""

    class Meta:
        model = Project
        fields = [
            'id',
            'title',
            'description',
            'created_at',
            'updated_at',
            'organization',
            'total_predictions_number',
            'is_published',
            'model_version',
            'is_draft',
            'ml_type_name',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class PredictionResultSerializer(serializers.ModelSerializer):
    """Serializer for individual prediction results."""
    
    class Meta:
        model = PredictionResult
        fields = ['id', 'label', 'score', 'created_at']


class PredictionSerializer(serializers.ModelSerializer):
    """Serializer for prediction data with results."""
    results = PredictionResultSerializer(many=True, read_only=True)

    class Meta:
        model = Prediction
        fields = [
            'id', 'image_url',
            'predicted_at', 'results',
            'created_at', 'updated_at'
        ]


class ProjectDetailSerializer(serializers.ModelSerializer):
    """Serializer for detailed project information with latest prediction."""
    labels = LabelSerializer(many=True, read_only=True)
    prediction = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = [
            'id',
            'title',
            'description',
            'created_at',
            'updated_at',
            'organization',
            'total_predictions_number',
            'is_published',
            'model_version',
            'is_draft',
            'ml_type_name',
            'labels',
            'prediction',
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'labels', 'prediction']

    def get_prediction(self, obj):
        """Get the latest prediction for the project."""
        newest_prediction = (
            Prediction.objects
            .filter(ls_project_id=obj.ls_project_id)
            .order_by('-predicted_at')
            .prefetch_related('results')
            .first()
        )
        
        if newest_prediction:
            return PredictionSerializer(newest_prediction).data
        return None
