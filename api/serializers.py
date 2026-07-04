# api/serializers.py
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import PracticeSession, UserProfile, Script

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'first_name', 'last_name']
    
    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        UserProfile.objects.create(user=user)
        return user

class SessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PracticeSession
        fields = ['id', 'created_at', 'accuracy', 'grade', 'grade_label',
                  'total_words', 'flagged_words', 'skipped_words', 'result_json']
        read_only_fields = fields

class UserProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    email = serializers.EmailField(source='user.email', read_only=True)
    class Meta:
        model = UserProfile
        fields = ['username', 'email', 'proficiency_level', 'reference_accent',
                  'voice_feedback', 'total_sessions', 'best_score']

# ADD THIS SERIALIZER
class ScriptSerializer(serializers.ModelSerializer):
    word_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Script
        fields = ['id', 'title', 'description', 'level', 'mode', 'content',
                  'is_builtin', 'created_at', 'word_count']
        read_only_fields = ['id', 'created_at', 'is_builtin']
    
    def get_word_count(self, obj):
        return obj.word_count()