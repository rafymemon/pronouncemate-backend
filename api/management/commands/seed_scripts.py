# api/management/commands/seed_scripts.py

from django.core.management.base import BaseCommand
from api.models import Script


SCRIPTS = [

    # ══ PUBLIC SPEAKING — BEGINNER ════════════════════════════
    {
        "title":       "Good Morning Introduction",
        "description": "Simple opening speech for beginners",
        "mode":        "public_speaking",
        "level":       "beginner",
        "content": (
            "Good morning everyone. My name is Ahmed. "
            "I am happy to be here today. "
            "Thank you for giving me this opportunity to speak. "
            "I hope you enjoy my talk."
        ),
    },
    {
        "title":       "Talking About Yourself",
        "description": "Basic self-introduction for new speakers",
        "mode":        "public_speaking",
        "level":       "beginner",
        "content": (
            "Hello. My name is Sara and I am a student. "
            "I study at Mehran University. "
            "I like reading books and learning new things. "
            "I want to improve my English speaking skills. "
            "Thank you for listening to me."
        ),
    },

    # ══ PUBLIC SPEAKING — INTERMEDIATE ════════════════════════
    {
        "title":       "Benefits of Technology",
        "description": "Short speech about technology in everyday life",
        "mode":        "public_speaking",
        "level":       "intermediate",
        "content": (
            "Technology has changed the way we live and work. "
            "Today we use smartphones, computers, and the internet every day. "
            "These tools help us communicate faster and learn more easily. "
            "However, we must use technology responsibly. "
            "We should make sure it helps us rather than distracts us. "
            "Thank you."
        ),
    },
    {
        "title":       "Conference Opening",
        "description": "Professional conference opening speech",
        "mode":        "public_speaking",
        "level":       "intermediate",
        "content": (
            "Hello. How are you today. "
            "It is great to see you again. "
            "I hope you are doing well and enjoying your day. "
            "Welcome to our annual conference on technology and innovation. "
            "We have an exciting program prepared for you today. "
            "I am confident you will find the sessions both informative and inspiring."
        ),
    },

    # ══ PUBLIC SPEAKING — UPPER INTERMEDIATE ══════════════════
    {
        "title":       "The Importance of Education",
        "description": "Persuasive speech on education for society",
        "mode":        "public_speaking",
        "level":       "upper_intermediate",
        "content": (
            "Education is the foundation of every successful society. "
            "When people have access to quality education, they can improve their lives "
            "and contribute meaningfully to their communities. "
            "In Pakistan, we must invest more in our schools and universities. "
            "We must ensure that every child, regardless of their background, "
            "has the opportunity to learn and grow. "
            "Education is not just a right. It is a responsibility we share as a nation."
        ),
    },
    {
        "title":       "Presenting a Project",
        "description": "Technical project presentation at university level",
        "mode":        "public_speaking",
        "level":       "upper_intermediate",
        "content": (
            "Good afternoon. Today I will present our final year project, PronounceMate. "
            "PronounceMate is an artificial intelligence system that helps learners "
            "improve their English pronunciation in real time. "
            "The system listens to the user's speech, identifies mispronounced words, "
            "and delivers immediate corrective feedback. "
            "We used Faster-Whisper for speech recognition and wav2vec for acoustic analysis. "
            "Our evaluation shows a Pearson Correlation Coefficient of above zero point six five "
            "against human expert scores on the SpeechOcean benchmark."
        ),
    },

    # ══ PUBLIC SPEAKING — ADVANCED ════════════════════════════
    {
        "title":       "Leadership and Responsibility",
        "description": "Formal speech on leadership principles",
        "mode":        "public_speaking",
        "level":       "advanced",
        "content": (
            "Distinguished guests, colleagues, and fellow students. "
            "True leadership is not about authority or position. "
            "It is about the willingness to serve, to listen, and to take responsibility "
            "when things go wrong rather than when things go right. "
            "The most effective leaders I have encountered throughout my career "
            "share one defining characteristic. "
            "They measure their success not by personal achievement "
            "but by the growth and development of the people around them. "
            "That is the kind of leadership our country needs today."
        ),
    },

    # ══ PUBLIC SPEAKING — FLUENT ══════════════════════════════
    {
        "title":       "Artificial Intelligence and Society",
        "description": "Advanced discourse on AI ethics and impact",
        "mode":        "public_speaking",
        "level":       "fluent",
        "content": (
            "As artificial intelligence continues to reshape the boundaries of what machines "
            "can accomplish, we find ourselves confronting profound questions "
            "about the future of human agency, creativity, and purpose. "
            "The promise of AI is extraordinary. "
            "Systems that can diagnose diseases with superhuman accuracy, "
            "predict climate patterns decades in advance, and accelerate scientific discovery "
            "across every discipline simultaneously. "
            "Yet the responsible development of these technologies demands "
            "that we proceed with both ambition and humility, "
            "recognizing that the consequences of our choices today "
            "will reverberate through generations we will never meet."
        ),
    },


    # ══ DAILY CONVERSATION — BEGINNER ═════════════════════════
    {
        "title":       "Morning Greetings",
        "description": "Simple everyday greetings and responses",
        "mode":        "daily_conversation",
        "level":       "beginner",
        "content": (
            "Good morning. How are you. "
            "I am fine thank you. "
            "Did you sleep well last night. "
            "Yes I slept very well. "
            "What are you doing today. "
            "I am going to the market. "
            "Have a good day."
        ),
    },
    {
        "title":       "Ordering Food",
        "description": "Conversation at a restaurant or cafe",
        "mode":        "daily_conversation",
        "level":       "beginner",
        "content": (
            "Can I see the menu please. "
            "I would like a cup of tea. "
            "Do you have anything to eat. "
            "Yes we have sandwiches and samosas. "
            "I will have a sandwich please. "
            "How much does it cost. "
            "It costs one hundred rupees. "
            "Thank you very much."
        ),
    },

    # ══ DAILY CONVERSATION — INTERMEDIATE ═════════════════════
    {
        "title":       "At the University",
        "description": "Common conversations between students",
        "mode":        "daily_conversation",
        "level":       "intermediate",
        "content": (
            "Have you finished the assignment yet. "
            "Not yet. I am still working on it. "
            "When is it due. "
            "It is due tomorrow morning. "
            "I think we should work on it together. "
            "That is a good idea. "
            "Let us meet in the library at five o'clock. "
            "Sure. I will see you there."
        ),
    },
    {
        "title":       "Making Plans",
        "description": "Arranging to meet a friend or colleague",
        "mode":        "daily_conversation",
        "level":       "intermediate",
        "content": (
            "Are you free this weekend. "
            "I think so. What did you have in mind. "
            "I was thinking we could go to the park. "
            "That sounds great. What time were you thinking. "
            "How about Saturday afternoon around three o'clock. "
            "That works for me. Should I bring anything. "
            "Just bring yourself and your good mood. "
            "Perfect. I am looking forward to it."
        ),
    },

    # ══ DAILY CONVERSATION — UPPER INTERMEDIATE ═══════════════
    {
        "title":       "Discussing the News",
        "description": "Conversation about current events",
        "mode":        "daily_conversation",
        "level":       "upper_intermediate",
        "content": (
            "Did you see the news this morning. "
            "No I did not. What happened. "
            "There was a major announcement about the new education policy. "
            "Really. What did they say. "
            "The government plans to increase funding for public universities by thirty percent. "
            "That is actually quite significant. "
            "I hope it leads to real improvements in the quality of education. "
            "I agree. We have been waiting for this kind of investment for a long time."
        ),
    },

    # ══ DAILY CONVERSATION — ADVANCED ═════════════════════════
    {
        "title":       "Career Discussion",
        "description": "Professional conversation about career goals",
        "mode":        "daily_conversation",
        "level":       "advanced",
        "content": (
            "I have been thinking seriously about my career path lately. "
            "What specifically has been on your mind. "
            "I am trying to decide between pursuing a master's degree immediately "
            "or gaining some industry experience first. "
            "Both have their merits. What does your gut tell you. "
            "Honestly, I think the practical experience would make me a stronger candidate "
            "for a competitive graduate program later. "
            "That is a very mature way of looking at it. "
            "Have you spoken to anyone in the field about their perspective."
        ),
    },

    # ══ DAILY CONVERSATION — FLUENT ═══════════════════════════
    {
        "title":       "Philosophical Discussion",
        "description": "Deep intellectual conversation between friends",
        "mode":        "daily_conversation",
        "level":       "fluent",
        "content": (
            "I have been reading about the concept of free will lately "
            "and I find myself increasingly skeptical about whether it actually exists. "
            "That is a fascinating rabbit hole. "
            "What specifically challenged your intuitions. "
            "The neuroscience research suggesting that our brains make decisions "
            "several hundred milliseconds before we become consciously aware of them. "
            "That is compelling but I would push back on the interpretation. "
            "Consciousness might not need to initiate the action to still meaningfully participate in it. "
            "That is an interesting distinction. Say more."
        ),
    },
]


class Command(BaseCommand):
    help = 'Seed built-in scripts for all modes and levels'

    def handle(self, *args, **kwargs):
        created = 0
        skipped = 0
        for data in SCRIPTS:
            obj, was_created = Script.objects.get_or_create(
                title=data['title'],
                mode=data['mode'],
                is_builtin=True,
                defaults={
                    'description': data['description'],
                    'level':       data['level'],
                    'content':     data['content'],
                    'is_builtin':  True,
                }
            )
            if was_created:
                created += 1
                self.stdout.write(
                    self.style.SUCCESS(f'  ✓ {obj.mode} / {obj.level} / {obj.title}'))
            else:
                skipped += 1

        self.stdout.write(self.style.SUCCESS(
            f'\nDone — {created} created, {skipped} already existed.'))