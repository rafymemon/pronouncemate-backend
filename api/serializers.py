# api/serializers.py
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import PracticeSession, UserProfile, Script
from .models import (
    PracticeSession, UserProfile, Script,
    WordMastery, LevelProgressionStatus, FocusScript,
)


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


class ScriptSerializer(serializers.ModelSerializer):
    word_count = serializers.SerializerMethodField()
    preview    = serializers.SerializerMethodField()

    class Meta:
        model  = Script
        fields = [
            'id', 'title', 'description',
            'level', 'mode', 'content',
            'is_builtin', 'word_count', 'preview',
        ]

    def get_word_count(self, obj):
        return len(obj.content.split())

    def get_preview(self, obj):
        words = obj.content.split()
        preview = ' '.join(words[:18])
        return preview + '...' if len(words) > 18 else preview





class RotatedScriptSerializer(serializers.ModelSerializer):
    word_count = serializers.SerializerMethodField()
    preview    = serializers.SerializerMethodField()

    class Meta:
        model = Script
        fields = [
            'id', 'title', 'description', 'level', 'mode',
            'content', 'is_builtin', 'word_count', 'preview',
        ]

    def get_word_count(self, obj):
        return obj.word_count()

    def get_preview(self, obj):
        text = obj.content
        return text if len(text) <= 100 else text[:100] + '...'






class WordMasterySerializer(serializers.ModelSerializer):
    class Meta:
        model = WordMastery
        fields = [
            'id', 'word', 'error_type', 'consecutive_correct',
            'status', 'last_seen_at', 'mastered_at',
        ]
        read_only_fields = fields


class LevelProgressionStatusSerializer(serializers.ModelSerializer):
    gates = serializers.SerializerMethodField()

    class Meta:
        model = LevelProgressionStatus
        fields = [
            'current_level', 'rolling_avg_score', 'rolling_error_rate',
            'mastery_ratio', 'flagged_phoneme_rate',
            'eligible_for_promotion', 'last_evaluated_at', 'gates',
        ]
        read_only_fields = fields

    def get_gates(self, obj):
        """
        Individual pass/fail per criterion, for the "X / 5 criteria met"
        transparency panel on the dashboard (Improvement Phase doc, Section 2).
        """
        from .progression import (
            GATE_MIN_AVG_SCORE, GATE_MAX_ERROR_RATE,
            GATE_MIN_MASTERY_PCT, GATE_MAX_PHONEME_PCT, GATE_MIN_SESSIONS,
        )
        sessions_at_level = PracticeSession.objects.filter(
            user=obj.user, created_at__gte=obj.level_changed_at
        ).count()

        return {
            'sessions': {
                'passed': sessions_at_level >= GATE_MIN_SESSIONS,
                'current': sessions_at_level,
                'required': GATE_MIN_SESSIONS,
            },
            'score': {
                'passed': obj.rolling_avg_score >= GATE_MIN_AVG_SCORE,
                'current': obj.rolling_avg_score,
                'required': GATE_MIN_AVG_SCORE,
            },
            'error_rate': {
                'passed': obj.rolling_error_rate <= GATE_MAX_ERROR_RATE,
                'current': obj.rolling_error_rate,
                'required': GATE_MAX_ERROR_RATE,
            },
            'mastery': {
                'passed': obj.mastery_ratio >= GATE_MIN_MASTERY_PCT,
                'current': obj.mastery_ratio,
                'required': GATE_MIN_MASTERY_PCT,
            },
            'phoneme': {
                'passed': obj.flagged_phoneme_rate <= GATE_MAX_PHONEME_PCT,
                'current': obj.flagged_phoneme_rate,
                'required': GATE_MAX_PHONEME_PCT,
            },
        }


class FocusScriptSerializer(serializers.ModelSerializer):
    class Meta:
        model = FocusScript
        fields = [
            'id', 'generated_at', 'source_words', 'regression_words', 'script_text',
            'completed', 'completed_at', 'pre_error_rate', 'post_error_rate',
        ]
        read_only_fields = fields