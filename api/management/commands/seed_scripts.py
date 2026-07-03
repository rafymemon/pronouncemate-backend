# api/management/commands/seed_scripts.py
import os
from django.core.management.base import BaseCommand
from api.models import Script


BUILTIN_SCRIPTS = [

    # ── BEGINNER ──────────────────────────────────────────────────────────
    {
        "title":       "Daily Greetings",
        "description": "Common phrases used every day — perfect for beginners",
        "level":       "beginner",
        "mode":        "daily",
        "content":     (
            "Good morning. How are you today? "
            "I am doing very well thank you. "
            "The weather is nice today. "
            "I would like a cup of tea please. "
            "Have a good day."
        ),
    },
    {
        "title":       "At the Market",
        "description": "Shopping conversation for everyday practice",
        "level":       "beginner",
        "mode":        "daily",
        "content":     (
            "How much does this cost? "
            "That is very expensive. "
            "Can you give me a better price? "
            "I will think about it. "
            "Thank you very much."
        ),
    },

    # ── INTERMEDIATE ──────────────────────────────────────────────────────
    {
        "title":       "Introducing Yourself",
        "description": "Professional self-introduction for interviews and meetings",
        "level":       "intermediate",
        "mode":        "formal",
        "content":     (
            "My name is Ahmed and I am a software engineering student at MUET. "
            "I am particularly interested in artificial intelligence and mobile application development. "
            "I have experience working with Python, Flutter, and Django. "
            "I am looking for opportunities to apply my skills in a professional environment. "
            "I am a quick learner and I work well in a team."
        ),
    },
    {
        "title":       "Describing Your Project",
        "description": "Technical explanation suitable for viva and presentations",
        "level":       "intermediate",
        "mode":        "public",
        "content":     (
            "Our system is designed to improve English pronunciation for Pakistani learners. "
            "The application listens to the user reading a script and identifies mispronounced words. "
            "We use artificial intelligence models to compare the user's speech with the correct pronunciation. "
            "Immediate feedback is provided through both audio and text. "
            "The system is free, works offline, and runs on any Android or iOS device."
        ),
    },

    # ── ADVANCED ──────────────────────────────────────────────────────────
    {
        "title":       "Technology and Society",
        "description": "Extended passage for advanced fluency and pronunciation practice",
        "level":       "advanced",
        "mode":        "public",
        "content":     (
            "Artificial intelligence is transforming the way we communicate, learn, and work. "
            "In education particularly, intelligent systems are enabling personalized learning "
            "at a scale that was previously impossible. "
            "Students in remote areas can now access the same quality of instruction "
            "as those in well-resourced urban centers. "
            "However, the responsible development of these technologies requires "
            "careful consideration of privacy, equity, and the long-term impact on human skills. "
            "As future engineers, it is our responsibility to build systems that benefit everyone."
        ),
    },
]


class Command(BaseCommand):
    help = 'Seed database with built-in practice scripts'

    def handle(self, *args, **kwargs):
        created = 0
        skipped = 0

        for data in BUILTIN_SCRIPTS:
            obj, was_created = Script.objects.get_or_create(
                title=data['title'],
                is_builtin=True,
                defaults={
                    'description': data['description'],
                    'level':       data['level'],
                    'mode':        data['mode'],
                    'content':     data['content'],
                    'is_builtin':  True,
                }
            )
            if was_created:
                created += 1
                self.stdout.write(self.style.SUCCESS(f"  Created: {obj.title}"))
            else:
                skipped += 1
                self.stdout.write(f"  Skipped (exists): {obj.title}")

        self.stdout.write(
            self.style.SUCCESS(
                f"\nDone — {created} scripts created, {skipped} already existed."
            )
        )