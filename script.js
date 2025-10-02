document.addEventListener('DOMContentLoaded', function() {
    // Smooth scrolling for navigation links
    document.querySelectorAll('nav a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();

            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Optional: Add a simple animation on scroll for sections
    const sections = document.querySelectorAll('section');

    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    sections.forEach(section => {
        section.classList.add('hidden'); // Add a class to hide initially
        observer.observe(section);
    });

    // Add CSS for .hidden and .fade-in to style.css if not already there
    // .hidden { opacity: 0; transform: translateY(20px); transition: opacity 0.6s ease-out, transform 0.6s ease-out; }
    // .fade-in { opacity: 1; transform: translateY(0); }
});

