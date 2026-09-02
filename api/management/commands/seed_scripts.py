"""
api/management/commands/seed_scripts.py

Built-in script library, consolidated to 3 proficiency levels:
beginner, intermediate, expert — 10 scripts per level, per mode
(60 total: 30 public_speaking + 30 daily_conversation).

Consolidation from the previous 5-level library:
    - beginner            -> beginner   (5 originals kept + 5 new added)
    - intermediate         -> intermediate
    - upper_intermediate    -> intermediate (relabeled, content unchanged)
    - advanced              -> expert       (relabeled, content unchanged)
    - fluent                -> expert       (relabeled, content unchanged)

This REPLACES the previous seed_scripts.py. Uses get_or_create on title
so re-running this command is safe and won't create duplicates. Note:
if you're re-seeding an existing database, the old upper_intermediate/
advanced/fluent rows will now read "intermediate"/"expert" here, but any
already-created rows in the DB keep whatever `level` they were created
with — update those rows (or wipe and re-seed) if you need the level
field itself to change on existing data.

Run with:
    python manage.py seed_scripts
"""

from django.core.management.base import BaseCommand
from api.models import Script

SCRIPTS = [
    # ══════════════════════════════════════════════════════════════
    # PUBLIC SPEAKING — BEGINNER
    # ══════════════════════════════════════════════════════════════
    {
        "title": "Introducing Myself",
        "description": "A simple self-introduction for a classroom or small group.",
        "mode": "public_speaking",
        "level": "beginner",
        "content": (
            "Good morning, everyone. My name is Ali and I am a student at this "
            "university. I am studying Software Engineering because I enjoy "
            "solving problems and building new things. In my free time, I like "
            "to read books and play cricket with my friends. I am happy to be "
            "here today, and I look forward to learning with all of you."
        ),
    },
    {
        "title": "My Favorite Hobby",
        "description": "A short, friendly talk about a personal hobby.",
        "mode": "public_speaking",
        "level": "beginner",
        "content": (
            "Today I want to talk about my favorite hobby, which is painting. "
            "I started painting when I was ten years old. I love using bright "
            "colors to create simple pictures of nature. Painting helps me "
            "relax after a long day of classes. I think everyone should have a "
            "hobby that makes them happy."
        ),
    },
    {
        "title": "A Short Thank You Speech",
        "description": "A brief speech to thank people after receiving help or a gift.",
        "mode": "public_speaking",
        "level": "beginner",
        "content": (
            "Thank you all very much for coming here today. I am truly grateful "
            "for your support and kindness. This means a lot to me, and I will "
            "always remember this moment. I want to thank my teachers and my "
            "friends for always helping me. Thank you again, and have a "
            "wonderful day."
        ),
    },
    {
        "title": "Welcome to Our Event",
        "description": "A short welcome address for a small campus event.",
        "mode": "public_speaking",
        "level": "beginner",
        "content": (
            "Good afternoon, and welcome to our event. We are very happy to "
            "see so many of you here today. This event was planned by our "
            "student club, and we hope you enjoy it. Please feel free to ask "
            "questions and meet new people. Let us begin the program now."
        ),
    },
    {
        "title": "My Daily Routine",
        "description": "A simple, present-tense description of a typical day.",
        "mode": "public_speaking",
        "level": "beginner",
        "content": (
            "Let me tell you about my daily routine. I wake up early every "
            "morning and drink a cup of tea. Then I get ready and walk to my "
            "university. My classes usually start at nine o'clock. After "
            "class, I study in the library for two hours. In the evening, I "
            "relax and spend time with my family."
        ),
    },
    {
        "title": "My Favorite Food",
        "description": "A simple talk describing a favorite dish and why it matters.",
        "mode": "public_speaking",
        "level": "beginner",
        "content": (
            "Today I want to talk about my favorite food, which is biryani. I "
            "enjoy biryani because it has a wonderful mix of spices and rice. "
            "My mother cooks it every Sunday for our family lunch. The smell "
            "of biryani cooking always makes me feel happy and hungry. I "
            "think food can bring people together in a special way. I hope "
            "you all get to try a good plate of biryani someday."
        ),
    },
    {
        "title": "My Best Friend",
        "description": "A short, warm talk introducing a close friend.",
        "mode": "public_speaking",
        "level": "beginner",
        "content": (
            "I would like to talk about my best friend, Hassan. We have known "
            "each other since primary school. Hassan is always kind and ready "
            "to help me when I need support. We enjoy playing football "
            "together every weekend. A good friend makes difficult days feel "
            "much easier. I am very grateful to have Hassan as my friend."
        ),
    },
    {
        "title": "A Visit to the Park",
        "description": "A simple recount of a relaxing family outing.",
        "mode": "public_speaking",
        "level": "beginner",
        "content": (
            "Last weekend, I visited a park near my house with my family. The "
            "park had many green trees and a small lake in the middle. We sat "
            "on the grass and had a simple picnic together. My little sister "
            "enjoyed feeding the ducks near the lake. Spending time outdoors "
            "always makes me feel calm and refreshed. I hope to visit the "
            "park again very soon."
        ),
    },
    {
        "title": "My Weekend Plans",
        "description": "A short, future-tense talk about upcoming weekend activities.",
        "mode": "public_speaking",
        "level": "beginner",
        "content": (
            "Let me share my plans for this weekend. On Saturday morning, I "
            "will clean my room and finish my homework. In the afternoon, I "
            "plan to meet my cousins for lunch. On Sunday, I want to relax "
            "and watch a movie at home. I also hope to call my grandmother "
            "and ask about her health. Weekends are important because they "
            "give me time to rest."
        ),
    },
    {
        "title": "Why I Like My City",
        "description": "A simple, positive talk about one's hometown or city.",
        "mode": "public_speaking",
        "level": "beginner",
        "content": (
            "I want to talk about why I like living in my city. My city has "
            "friendly people and many good places to visit. There are nice "
            "parks, markets, and restaurants close to my home. I especially "
            "enjoy walking around the old streets in the evening. The food "
            "here is also very delicious and affordable. I feel proud and "
            "comfortable calling this city my home."
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # PUBLIC SPEAKING — INTERMEDIATE
    # ══════════════════════════════════════════════════════════════
    {
        "title": "The Importance of Time Management",
        "description": "A short talk on managing time effectively as a student.",
        "mode": "public_speaking",
        "level": "intermediate",
        "content": (
            "Time management is one of the most important skills a student can "
            "develop. Many of us struggle to balance classes, assignments, and "
            "personal life. By creating a simple daily schedule, we can reduce "
            "stress and get more done. It also helps to set clear priorities "
            "and avoid distractions like excessive phone use. When we manage "
            "our time well, we perform better in our studies and still find "
            "time to relax."
        ),
    },
    {
        "title": "Why I Chose My Field of Study",
        "description": "Explaining personal motivation for an academic path.",
        "mode": "public_speaking",
        "level": "intermediate",
        "content": (
            "I chose to study Software Engineering because I have always been "
            "curious about how applications and websites are built. During my "
            "final year of school, I built a small project that solved a real "
            "problem for my neighborhood, and that experience changed my "
            "perspective. I realized that technology can genuinely improve "
            "people's lives. Since then, I have worked hard to strengthen my "
            "programming skills and understand how systems work together."
        ),
    },
    {
        "title": "The Benefits of Reading Books",
        "description": "A talk highlighting why reading remains valuable today.",
        "mode": "public_speaking",
        "level": "intermediate",
        "content": (
            "Reading books offers benefits that go far beyond entertainment. "
            "It expands our vocabulary, improves concentration, and exposes us "
            "to new ideas and perspectives. Unlike scrolling through social "
            "media, reading a book requires sustained attention, which trains "
            "our mind to focus for longer periods. Many successful people "
            "credit reading as a habit that shaped their thinking. I encourage "
            "everyone to set aside even fifteen minutes a day for reading."
        ),
    },
    {
        "title": "How Technology Changed Communication",
        "description": "Reflecting on the shift from traditional to digital communication.",
        "mode": "public_speaking",
        "level": "intermediate",
        "content": (
            "Communication has changed dramatically over the past two decades. "
            "We once relied on letters and landline phones to stay in touch "
            "with family and friends. Today, messaging apps and video calls "
            "allow us to connect instantly with people around the world. This "
            "shift has made communication faster and more convenient, but it "
            "has also changed how deeply we connect with one another. I "
            "believe we should use technology wisely to strengthen, not "
            "replace, real relationships."
        ),
    },
    {
        "title": "The Value of Teamwork",
        "description": "A talk on why collaboration matters in academic and work life.",
        "mode": "public_speaking",
        "level": "intermediate",
        "content": (
            "Teamwork plays a vital role in both university projects and "
            "future careers. When people with different strengths work "
            "together, they can solve problems more creatively than any "
            "individual working alone. Good teamwork requires clear "
            "communication, patience, and respect for other people's ideas. "
            "During my final year project, I learned that listening carefully "
            "to my teammates often led to better solutions than my own "
            "original plan."
        ),
    },
    {
        "title": "Balancing Studies and Social Life",
        "description": "Discussing the challenge of maintaining balance in university life.",
        "mode": "public_speaking",
        "level": "intermediate",
        "content": (
            "One of the biggest challenges university students face is finding "
            "a healthy balance between academic responsibilities and social "
            "life. On one hand, our coursework demands significant time and "
            "focus, especially during exam season. On the other hand, "
            "friendships and social activities are essential for our mental "
            "wellbeing and personal growth. I have found that setting clear "
            "boundaries around study time, while still making room for "
            "meaningful conversations with friends, helps me stay both "
            "productive and genuinely happy."
        ),
    },
    {
        "title": "The Role of Youth in Nation Building",
        "description": "A speech on the responsibility of young people in national development.",
        "mode": "public_speaking",
        "level": "intermediate",
        "content": (
            "Young people carry immense responsibility when it comes to "
            "shaping the future of their nation. We are the generation that "
            "will inherit the systems, institutions, and challenges left "
            "behind by those before us. Rather than waiting for opportunities "
            "to arrive, we must actively pursue education, innovation, and "
            "civic participation. Whether through entrepreneurship, community "
            "service, or simply voting responsibly, every young person has a "
            "role to play in building a stronger, more equitable society."
        ),
    },
    {
        "title": "Overcoming Fear of Public Speaking",
        "description": "Addressing a common anxiety and offering practical strategies.",
        "mode": "public_speaking",
        "level": "intermediate",
        "content": (
            "Public speaking is consistently ranked among people's greatest "
            "fears, often above the fear of death itself. This fear typically "
            "stems from a concern about being judged or making mistakes in "
            "front of an audience. However, this anxiety can be managed "
            "through consistent practice and preparation. Rehearsing your "
            "speech multiple times, focusing on your message rather than "
            "yourself, and starting with smaller audiences can gradually build "
            "your confidence. Remember that even experienced speakers still "
            "feel nervous before they begin."
        ),
    },
    {
        "title": "The Impact of Social Media on Society",
        "description": "Examining both the benefits and drawbacks of social media use.",
        "mode": "public_speaking",
        "level": "intermediate",
        "content": (
            "Social media has fundamentally transformed the way society "
            "communicates, consumes information, and forms opinions. On the "
            "positive side, it has democratized access to information and "
            "given a voice to communities that were previously unheard. "
            "However, it has also contributed to the spread of misinformation "
            "and has been linked to rising rates of anxiety among young "
            "users. As responsible citizens, we must learn to critically "
            "evaluate the content we consume rather than accepting it at face "
            "value."
        ),
    },
    {
        "title": "Preparing for a Successful Career",
        "description": "Advice on developing skills beyond academic knowledge.",
        "mode": "public_speaking",
        "level": "intermediate",
        "content": (
            "Academic knowledge alone is no longer sufficient to guarantee a "
            "successful career in today's competitive job market. Employers "
            "increasingly value practical skills such as effective "
            "communication, problem-solving, and adaptability to change. "
            "Building a strong professional network, gaining hands-on "
            "experience through internships, and continuously updating one's "
            "skill set are equally important. I encourage every student to "
            "step outside their comfort zone and seek opportunities that "
            "challenge their existing abilities."
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # PUBLIC SPEAKING — EXPERT
    # ══════════════════════════════════════════════════════════════
    {
        "title": "Leadership in Challenging Times",
        "description": "Exploring qualities of effective leadership during adversity.",
        "mode": "public_speaking",
        "level": "expert",
        "content": (
            "True leadership is rarely tested during periods of stability; it "
            "reveals itself most clearly in moments of crisis and uncertainty. "
            "Effective leaders during challenging times share several "
            "distinguishing qualities: they communicate transparently even "
            "when the news is unfavorable, they remain calm under pressure "
            "without dismissing the gravity of the situation, and they "
            "empower their teams rather than centralizing all decision-making "
            "authority. History demonstrates that organizations and nations "
            "which thrive through adversity are typically led by individuals "
            "who prioritize collective resilience over personal reputation."
        ),
    },
    {
        "title": "The Ethics of Artificial Intelligence",
        "description": "A discussion of the moral considerations surrounding AI development.",
        "mode": "public_speaking",
        "level": "expert",
        "content": (
            "As artificial intelligence systems become increasingly embedded "
            "in critical decision-making processes, from hiring practices to "
            "criminal justice, the ethical implications of these technologies "
            "demand urgent scrutiny. Algorithmic bias, often inherited from "
            "historical data, can perpetuate and even amplify existing social "
            "inequalities if left unchecked. Furthermore, questions of "
            "accountability become complicated when autonomous systems make "
            "consequential decisions without direct human oversight. "
            "Developers and policymakers alike must prioritize transparency, "
            "fairness, and rigorous auditing throughout the AI development "
            "lifecycle."
        ),
    },
    {
        "title": "Sustainable Development and Our Responsibility",
        "description": "A speech on environmental responsibility and long-term thinking.",
        "mode": "public_speaking",
        "level": "expert",
        "content": (
            "Sustainable development requires us to fundamentally reconsider "
            "the relationship between economic growth and environmental "
            "preservation. For decades, industrial progress was measured "
            "almost exclusively by short-term financial gains, with little "
            "regard for the long-term ecological consequences of resource "
            "extraction and consumption. Today, we understand that genuine "
            "prosperity cannot be sustained on a deteriorating planet. Each "
            "of us bears responsibility, whether through individual "
            "consumption choices, professional decisions, or civic advocacy, "
            "to demand policies that balance economic ambition with "
            "environmental stewardship."
        ),
    },
    {
        "title": "The Future of Higher Education",
        "description": "Examining how universities must adapt to a changing world.",
        "mode": "public_speaking",
        "level": "expert",
        "content": (
            "Higher education stands at a pivotal crossroads, challenged by "
            "the rapid pace of technological change and shifting labor market "
            "demands. Traditional models that emphasize rote memorization and "
            "standardized curricula are increasingly misaligned with a world "
            "that requires adaptability, critical thinking, and interdisciplinary "
            "problem-solving. Universities must evolve to incorporate "
            "experiential learning, closer industry collaboration, and "
            "flexible credentialing systems that recognize skills acquired "
            "outside conventional classrooms. Institutions that fail to adapt "
            "risk becoming irrelevant to the very students they aim to serve."
        ),
    },
    {
        "title": "Bridging the Gap Between Theory and Practice",
        "description": "Discussing the disconnect between academic learning and real-world application.",
        "mode": "public_speaking",
        "level": "expert",
        "content": (
            "A persistent criticism of academic institutions is the gap "
            "between theoretical knowledge and its practical application in "
            "professional settings. Students often graduate with strong "
            "conceptual foundations but limited experience translating that "
            "knowledge into tangible solutions for real-world problems. "
            "Closing this gap requires deliberate curriculum design that "
            "integrates hands-on projects, industry mentorship, and iterative "
            "feedback from practitioners. Educational institutions that "
            "successfully bridge this divide produce graduates who are not "
            "only knowledgeable but genuinely prepared to contribute from "
            "their very first day in the workforce."
        ),
    },
    {
        "title": "The Philosophy of Innovation",
        "description": "A reflective talk on the nature and drivers of innovation.",
        "mode": "public_speaking",
        "level": "expert",
        "content": (
            "Innovation is often romanticized as a sudden flash of genius, "
            "yet a closer examination of history reveals a far more nuanced "
            "and iterative process. Genuine breakthroughs typically emerge "
            "from the patient accumulation of incremental insights, "
            "punctuated by moments of unconventional thinking that challenge "
            "prevailing assumptions. What distinguishes truly innovative "
            "minds is not necessarily superior intelligence, but rather a "
            "tolerance for ambiguity and a willingness to fail repeatedly in "
            "pursuit of an uncertain outcome. As we cultivate environments "
            "that nurture innovation, we must resist the temptation to "
            "reward only polished successes, and instead recognize the "
            "immense value embedded in thoughtful, well-examined failure."
        ),
    },
    {
        "title": "Globalization and Cultural Identity",
        "description": "Exploring the tension between global integration and local identity.",
        "mode": "public_speaking",
        "level": "expert",
        "content": (
            "Globalization has ushered in unprecedented interconnectedness, "
            "enabling the free flow of goods, ideas, and people across "
            "borders that once seemed impermeable. Yet this very "
            "interconnectedness has provoked anxieties surrounding the "
            "erosion of distinct cultural identities in favor of a "
            "homogenized global culture, frequently dominated by Western "
            "commercial and media influences. The challenge before us is not "
            "to resist globalization outright, an increasingly implausible "
            "proposition, but rather to cultivate what might be termed "
            "'rooted cosmopolitanism': an openness to global exchange that "
            "remains firmly anchored in the preservation of local traditions, "
            "languages, and values."
        ),
    },
    {
        "title": "The Paradox of Choice in Modern Life",
        "description": "A talk examining how abundant options can paradoxically reduce wellbeing.",
        "mode": "public_speaking",
        "level": "expert",
        "content": (
            "Contemporary society operates under the assumption that greater "
            "choice invariably correlates with greater freedom and, by "
            "extension, greater happiness. Yet psychological research "
            "consistently suggests a counterintuitive phenomenon: beyond a "
            "certain threshold, an abundance of options can induce paralysis, "
            "heightened anxiety, and diminished satisfaction with whatever "
            "choice is ultimately made. This paradox manifests not only in "
            "consumer decisions, but in career paths, relationships, and even "
            "educational trajectories. Navigating this paradox demands a "
            "deliberate cultivation of what philosophers call 'satisficing', "
            "the discipline of choosing what is sufficiently good rather than "
            "exhaustively pursuing an elusive optimum."
        ),
    },
    {
        "title": "Redefining Success in the Digital Age",
        "description": "Interrogating traditional metrics of success amid technological change.",
        "mode": "public_speaking",
        "level": "expert",
        "content": (
            "For generations, success was measured through relatively "
            "static benchmarks: stable employment, homeownership, and "
            "incremental career advancement within a single organization. "
            "The digital age has fundamentally destabilized these metrics, "
            "giving rise to gig economies, remote entrepreneurship, and "
            "career trajectories that defy linear progression entirely. This "
            "shift compels us to interrogate whether our inherited "
            "definitions of success remain applicable, or whether we must "
            "construct more personalized, values-driven frameworks that "
            "prioritize autonomy, purpose, and adaptability over conventional "
            "markers of stability and prestige."
        ),
    },
    {
        "title": "The Intersection of Science and Ethics",
        "description": "A nuanced talk on the moral responsibilities accompanying scientific progress.",
        "mode": "public_speaking",
        "level": "expert",
        "content": (
            "Scientific inquiry has historically been justified by its "
            "pursuit of objective truth, ostensibly detached from moral "
            "considerations that govern other domains of human activity. Yet "
            "the twentieth and twenty-first centuries have repeatedly "
            "demonstrated that scientific capability divorced from ethical "
            "reflection can yield consequences of profound and sometimes "
            "irreversible magnitude, from nuclear weaponry to genetic "
            "engineering. The scientific community bears an obligation not "
            "merely to ask what is technically possible, but to rigorously "
            "interrogate what is ethically permissible, cultivating "
            "interdisciplinary dialogue between researchers, ethicists, and "
            "the broader public whose lives are ultimately shaped by these "
            "advancements."
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # DAILY CONVERSATION — BEGINNER
    # ══════════════════════════════════════════════════════════════
    {
        "title": "Meeting a New Friend",
        "description": "A simple conversation for introducing yourself to someone new.",
        "mode": "daily_conversation",
        "level": "beginner",
        "content": (
            "Hi, my name is Sara. What is your name? Nice to meet you! Are "
            "you a new student here? I am also new, so maybe we can be "
            "friends. Do you like this university so far? I think the campus "
            "is very beautiful. Would you like to have lunch together "
            "tomorrow?"
        ),
    },
    {
        "title": "At the Grocery Store",
        "description": "A common everyday conversation while shopping for groceries.",
        "mode": "daily_conversation",
        "level": "beginner",
        "content": (
            "Excuse me, where can I find the milk? Thank you very much. How "
            "much does this bread cost? That is a bit expensive, do you have "
            "a cheaper one? I also need some fruits, please. Do you have "
            "fresh apples today? Great, I will take five apples and this "
            "bread."
        ),
    },
    {
        "title": "Asking for Directions",
        "description": "A short conversation for finding your way in a new place.",
        "mode": "daily_conversation",
        "level": "beginner",
        "content": (
            "Excuse me, can you help me? I am looking for the library. Is it "
            "far from here? Should I go straight or turn left? Thank you so "
            "much for your help. One more question, is there a coffee shop "
            "near the library too? Thank you, have a nice day."
        ),
    },
    {
        "title": "A Phone Call to a Friend",
        "description": "A casual phone conversation between two friends.",
        "mode": "daily_conversation",
        "level": "beginner",
        "content": (
            "Hello, how are you today? I am doing well, thank you. What are "
            "you doing right now? I am just relaxing at home. Do you want to "
            "meet later this evening? We can go to the park and talk. Okay, "
            "see you at six o'clock. Bye for now."
        ),
    },
    {
        "title": "Talking About the Weather",
        "description": "A light, everyday conversation about the weather.",
        "mode": "daily_conversation",
        "level": "beginner",
        "content": (
            "It is very hot today, isn't it? Yes, I heard it will rain "
            "tomorrow. I hope the weather gets cooler soon. Do you like "
            "summer or winter more? I prefer winter because I can wear warm "
            "clothes. What about you? I like summer because I can swim."
        ),
    },
    {
        "title": "Ordering Food at a Restaurant",
        "description": "A simple dialogue for ordering a meal.",
        "mode": "daily_conversation",
        "level": "beginner",
        "content": (
            "Hello, could I please see the menu? Thank you. What do you "
            "recommend for someone who likes spicy food? That sounds great, "
            "I will have that dish. Could I also get a glass of water, "
            "please? Thank you very much, I am looking forward to my meal."
        ),
    },
    {
        "title": "Talking About Pets",
        "description": "A friendly conversation about pets and animals.",
        "mode": "daily_conversation",
        "level": "beginner",
        "content": (
            "Do you have any pets at home? Yes, I have a small dog named "
            "Rocky. He is very playful and loves to run in the garden. What "
            "about you, do you have any pets? I have a cat, but she prefers "
            "sleeping most of the day. Pets really do make our homes feel "
            "warmer."
        ),
    },
    {
        "title": "A Conversation at the Bus Stop",
        "description": "A short, practical exchange while waiting for public transport.",
        "mode": "daily_conversation",
        "level": "beginner",
        "content": (
            "Excuse me, does this bus go to the city center? Yes, it does, "
            "but it might take about twenty minutes. Thank you, I was not "
            "sure if I was waiting at the right stop. No problem, the bus "
            "should arrive very soon. Do you also take this bus every day? "
            "Yes, I use it to go to work every morning."
        ),
    },
    {
        "title": "Making Plans for the Weekend",
        "description": "A simple back-and-forth about weekend plans between friends.",
        "mode": "daily_conversation",
        "level": "beginner",
        "content": (
            "What are you doing this weekend? I do not have any plans yet, "
            "why do you ask? I was thinking we could go watch a movie "
            "together. That sounds like a good idea, what time works for "
            "you? How about Saturday evening around seven o'clock? Perfect, "
            "I will see you then."
        ),
    },
    {
        "title": "Talking About School Subjects",
        "description": "A simple conversation comparing favorite and difficult school subjects.",
        "mode": "daily_conversation",
        "level": "beginner",
        "content": (
            "Which subject do you enjoy the most at school? I really like "
            "mathematics because I enjoy solving problems. What about you, "
            "what is your favorite subject? I prefer English because I love "
            "reading stories. Do you find any subject difficult? I sometimes "
            "find science a little challenging. Maybe we can study together "
            "sometime."
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # DAILY CONVERSATION — INTERMEDIATE
    # ══════════════════════════════════════════════════════════════
    {
        "title": "Planning a Weekend Trip",
        "description": "A conversation between friends organizing a short trip.",
        "mode": "daily_conversation",
        "level": "intermediate",
        "content": (
            "So, are we still planning to go to the mountains this weekend? "
            "I think it's a great idea, but we should check the weather "
            "forecast first. If it rains, we might want to postpone the "
            "trip. I can book the transportation if you handle the hotel "
            "reservation. We should also decide how much money to bring for "
            "food and activities. Let's finalize everything by tomorrow "
            "evening so we don't run out of time."
        ),
    },
    {
        "title": "A Conversation at the Bank",
        "description": "A practical dialogue for handling banking tasks.",
        "mode": "daily_conversation",
        "level": "intermediate",
        "content": (
            "Good morning, I would like to open a new savings account. Could "
            "you please explain the requirements? I also need to know about "
            "the minimum balance and any monthly fees. Is it possible to "
            "link this account to a mobile banking app? That would be very "
            "convenient for me. Thank you for explaining everything so "
            "clearly. I will bring the required documents tomorrow."
        ),
    },
    {
        "title": "Talking About Family",
        "description": "A warm conversation sharing details about family members.",
        "mode": "daily_conversation",
        "level": "intermediate",
        "content": (
            "Tell me a little about your family. How many siblings do you "
            "have? I have one older brother and one younger sister. My "
            "brother works as an engineer, and my sister is still in school. "
            "We are a close family, and we try to have dinner together every "
            "evening. What about your family? Do you also have a similar "
            "tradition?"
        ),
    },
    {
        "title": "Discussing a Favorite Movie",
        "description": "A casual chat about films and personal preferences.",
        "mode": "daily_conversation",
        "level": "intermediate",
        "content": (
            "Have you watched any good movies recently? I just finished "
            "watching a fantastic drama last night. The storyline was quite "
            "unpredictable, and the acting was excellent. I usually prefer "
            "movies with strong character development over ones that focus "
            "only on action. What kind of movies do you enjoy the most? "
            "Maybe we could watch one together this weekend."
        ),
    },
    {
        "title": "At the Doctor's Clinic",
        "description": "A dialogue for describing symptoms during a medical visit.",
        "mode": "daily_conversation",
        "level": "intermediate",
        "content": (
            "Good afternoon, doctor. I have been feeling unwell for the past "
            "two days. I have a mild fever and a persistent headache. I also "
            "feel quite tired even after resting. Should I be worried about "
            "these symptoms? Is there any medicine you would recommend? "
            "Thank you, I will follow your advice and come back if it does "
            "not improve."
        ),
    },
    {
        "title": "Negotiating a Price at the Market",
        "description": "A back-and-forth negotiation typical of local marketplaces.",
        "mode": "daily_conversation",
        "level": "intermediate",
        "content": (
            "This looks like good quality fabric, but the price seems a bit "
            "steep for my budget. Would you be willing to lower it slightly "
            "if I buy two pieces instead of one? I understand your costs "
            "have gone up, but I was hoping we could reach a fair "
            "compromise. If you can offer a small discount, I would be happy "
            "to become a regular customer. That sounds reasonable, let's "
            "settle on that price."
        ),
    },
    {
        "title": "A Conversation About Career Goals",
        "description": "Two friends discussing their professional ambitions.",
        "mode": "daily_conversation",
        "level": "intermediate",
        "content": (
            "Have you decided what kind of career path you want to pursue "
            "after graduation? I have been considering a few different "
            "options, but I keep going back and forth between industry and "
            "further studies. There are definitely advantages to gaining "
            "practical experience early on, but a graduate degree could also "
            "open doors to more specialized roles. I think it ultimately "
            "depends on what kind of work-life balance you're hoping to "
            "achieve in the long run."
        ),
    },
    {
        "title": "Discussing Current Events",
        "description": "A conversation touching on recent news and its implications.",
        "mode": "daily_conversation",
        "level": "intermediate",
        "content": (
            "Did you see the news about the new policy changes announced "
            "yesterday? I think it could have a significant impact on small "
            "businesses in the coming months. Some people are worried it "
            "might increase costs, while others believe it will create more "
            "opportunities in the long term. It's hard to predict exactly "
            "how things will unfold, but I think we should keep an eye on "
            "how the implementation actually plays out."
        ),
    },
    {
        "title": "Giving Advice to a Friend",
        "description": "A supportive conversation offering guidance during a difficult decision.",
        "mode": "daily_conversation",
        "level": "intermediate",
        "content": (
            "I heard you've been struggling to decide between the two job "
            "offers. Honestly, I think you should weigh not just the salary, "
            "but also the growth opportunities each company offers. Sometimes "
            "a slightly lower starting salary is worth it if the role gives "
            "you more valuable experience. Whatever you decide, just make "
            "sure it aligns with what you actually want for your future, not "
            "just what looks impressive to others."
        ),
    },
    {
        "title": "Talking About Cultural Differences",
        "description": "A thoughtful discussion comparing customs across cultures.",
        "mode": "daily_conversation",
        "level": "intermediate",
        "content": (
            "It's fascinating how different our cultures can be when it "
            "comes to something as simple as greeting people. In some "
            "places, a firm handshake is expected, while in others, a slight "
            "bow or a hand over the heart is more appropriate. I think "
            "understanding these small differences really helps when you're "
            "traveling or working with people from different backgrounds. It "
            "shows respect and genuine interest in their way of life."
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # DAILY CONVERSATION — EXPERT
    # ══════════════════════════════════════════════════════════════
    {
        "title": "A Job Interview Conversation",
        "description": "A realistic simulation of a professional job interview.",
        "mode": "daily_conversation",
        "level": "expert",
        "content": (
            "Thank you for taking the time to meet with me today. Could you "
            "walk me through a challenging project you worked on and how "
            "you approached it? I was particularly responsible for "
            "coordinating between the technical team and stakeholders who "
            "had conflicting priorities. I learned that transparent "
            "communication early in the process prevents most "
            "misunderstandings down the line. I believe that experience has "
            "prepared me well for a role that requires both technical skill "
            "and strong interpersonal judgment."
        ),
    },
    {
        "title": "Debating a Controversial Topic",
        "description": "A respectful but firm exchange of opposing viewpoints.",
        "mode": "daily_conversation",
        "level": "expert",
        "content": (
            "I understand your perspective, but I have to respectfully "
            "disagree with the conclusion you're drawing from that evidence. "
            "While it's true that the data shows a correlation, I don't "
            "think it necessarily proves causation in this case. There could "
            "be several underlying factors we haven't accounted for. I "
            "appreciate that we see this differently, and I think that's "
            "exactly why these conversations are valuable, they force us to "
            "examine our assumptions more critically."
        ),
    },
    {
        "title": "Discussing Work-Life Balance",
        "description": "A nuanced conversation about managing professional and personal demands.",
        "mode": "daily_conversation",
        "level": "expert",
        "content": (
            "Honestly, I've been finding it increasingly difficult to "
            "maintain a healthy boundary between my work and personal life, "
            "especially since I started working remotely. It's tempting to "
            "just keep checking emails even after office hours, but I've "
            "noticed it's affecting my overall wellbeing. I've started "
            "setting a firm cutoff time in the evenings, and surprisingly, "
            "my productivity during work hours has actually improved rather "
            "than declined."
        ),
    },
    {
        "title": "A Conversation About Personal Finance",
        "description": "A practical discussion on budgeting and long-term financial planning.",
        "mode": "daily_conversation",
        "level": "expert",
        "content": (
            "I've been trying to get more disciplined about saving a "
            "portion of my income every month, but it's harder than I "
            "expected. What has worked for me is automating the transfer to "
            "a separate savings account as soon as I get paid, so I'm not "
            "tempted to spend it first. It also helps to track where my "
            "money actually goes, because I was surprised how much I was "
            "spending on things I didn't even value that much."
        ),
    },
    {
        "title": "Resolving a Misunderstanding",
        "description": "A conversation working through a conflict between colleagues.",
        "mode": "daily_conversation",
        "level": "expert",
        "content": (
            "I think there might have been a misunderstanding about what I "
            "meant in that email, and I wanted to clarify it directly rather "
            "than let it create tension between us. When I mentioned the "
            "deadline concern, I wasn't questioning your effort, I was "
            "genuinely worried about whether we had enough support from the "
            "other team. I should have been clearer about that, and I "
            "appreciate you bringing it up so we could sort it out."
        ),
    },
    {
        "title": "A Philosophical Discussion Over Coffee",
        "description": "An informal but intellectually rich conversation between friends.",
        "mode": "daily_conversation",
        "level": "expert",
        "content": (
            "I've been thinking lately about whether we truly have as much "
            "control over our decisions as we assume, or whether most of "
            "what we consider 'choice' is really shaped by circumstances we "
            "never consciously selected. It's a bit unsettling to consider, "
            "honestly, because so much of how we judge ourselves and others "
            "hinges on the assumption of free will. Then again, even if our "
            "choices are influenced by countless invisible factors, the "
            "subjective experience of deliberating still feels meaningful in "
            "its own right, doesn't it?"
        ),
    },
    {
        "title": "Discussing Climate Change Solutions",
        "description": "A thoughtful conversation weighing different approaches to climate action.",
        "mode": "daily_conversation",
        "level": "expert",
        "content": (
            "What strikes me most about the climate debate is how often the "
            "conversation gets reduced to individual responsibility, when "
            "the reality is that systemic, policy-level change tends to "
            "yield far greater impact than personal lifestyle adjustments "
            "alone. That's not to say individual choices are meaningless, "
            "but I think we sometimes let corporations and policymakers off "
            "the hook by framing this primarily as a matter of personal "
            "virtue rather than collective, structural accountability."
        ),
    },
    {
        "title": "A Conversation About Ethics in Business",
        "description": "A nuanced dialogue on balancing profit motives with moral responsibility.",
        "mode": "daily_conversation",
        "level": "expert",
        "content": (
            "I find it genuinely difficult to reconcile the pressure "
            "companies face to maximize shareholder returns with the "
            "broader ethical obligations they arguably have toward "
            "employees, communities, and the environment. In theory, these "
            "goals aren't necessarily in conflict, but in practice, "
            "short-term financial incentives often win out over long-term "
            "responsible decision-making. I think the companies that "
            "genuinely succeed at both are the ones that treat ethical "
            "considerations as integral to their strategy, rather than as an "
            "afterthought or a public relations exercise."
        ),
    },
    {
        "title": "Reflecting on Life Choices",
        "description": "An introspective conversation about paths taken and not taken.",
        "mode": "daily_conversation",
        "level": "expert",
        "content": (
            "Looking back, I don't think I regret the decisions I made so "
            "much as I regret the amount of anxiety I attached to making "
            "them in the first place. There's a particular kind of paralysis "
            "that comes from treating every choice as irreversible and "
            "life-defining, when in reality, most paths are far more "
            "malleable than we give them credit for in the moment. I've "
            "come to appreciate that reflecting honestly on a decision "
            "matters more than whether it was, in some abstract sense, the "
            "'right' one."
        ),
    },
    {
        "title": "A Debate on Technology and Privacy",
        "description": "A layered conversation weighing convenience against personal privacy.",
        "mode": "daily_conversation",
        "level": "expert",
        "content": (
            "What concerns me about the current trajectory of data "
            "collection isn't necessarily any single instance of it, but "
            "rather the cumulative, almost invisible way it normalizes "
            "constant surveillance as simply the cost of participating in "
            "modern life. We're often told that if we have nothing to hide, "
            "we have nothing to fear, but I think that framing "
            "fundamentally misunderstands what privacy actually protects: "
            "not secrecy for its own sake, but the basic autonomy to exist "
            "without being constantly observed, profiled, and predicted."
        ),
    },
]


class Command(BaseCommand):
    help = "Seeds the database with built-in practice scripts (10 per level, per mode)."

    def handle(self, *args, **options):
        created_count = 0
        skipped_count = 0

        for script_data in SCRIPTS:
            obj, created = Script.objects.get_or_create(
                title=script_data["title"],
                defaults={
                    "description": script_data["description"],
                    "mode": script_data["mode"],
                    "level": script_data["level"],
                    "content": script_data["content"],
                    "is_builtin": True,
                },
            )
            if created:
                created_count += 1
            else:
                skipped_count += 1

        self.stdout.write(self.style.SUCCESS(
            f"Seeding complete: {created_count} scripts created, "
            f"{skipped_count} already existed and were skipped. "
            f"Total in library: {Script.objects.filter(is_builtin=True).count()}"
        ))