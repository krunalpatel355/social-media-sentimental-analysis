// static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    const submitBtn = document.getElementById('submitSearch');
    const searchOptions = document.getElementById('searchOptions');
    const searchInput = document.getElementById('searchInput');
    const optionButtons = document.getElementById('optionButtons');
    const selectedOptionsList = document.getElementById('selectedOptionsList');
    const simpleSearchBtn = document.getElementById('simpleSearchBtn');
    const advancedSearchForm = document.getElementById('advancedSearchForm');
    const searchResults = document.getElementById('searchResults');
    const copySearchIdBtn = document.getElementById('copySearchId');
    
    let selectedOptions = new Set();

    // Handle initial search submission
    if (submitBtn) {
        submitBtn.addEventListener('click', function() {
            const searchQuery = searchInput.value.trim();
            
            if (searchQuery) {
                // Get options from VES
                fetch('/get_options', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `query=${encodeURIComponent(searchQuery)}`
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        // Show the options section
                        searchOptions.classList.remove('hidden');
                        
                        // Create option buttons
                        optionButtons.innerHTML = data.options.map(option => 
                            `<button class="option-btn" data-option="${option}">${option}</button>`
                        ).join('');
                        
                        // Add click handlers to option buttons
                        document.querySelectorAll('.option-btn').forEach(btn => {
                            btn.addEventListener('click', function() {
                                this.classList.toggle('selected');
                                const option = this.dataset.option;
                                
                                if (this.classList.contains('selected')) {
                                    selectedOptions.add(option);
                                } else {
                                    selectedOptions.delete(option);
                                }
                                
                                updateSelectedOptionsList();
                            });
                        });
                    }
                });
            }
        });
    }

    // Handle tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            // Update active tab button
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            // Show corresponding tab content
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById(`${this.dataset.tab}SearchTab`).classList.add('active');
        });
    });

    // Update selected options list
    function updateSelectedOptionsList() {
        selectedOptionsList.textContent = Array.from(selectedOptions).join(', ');
    }

    // Handle simple search
    if (simpleSearchBtn) {
        simpleSearchBtn.addEventListener('click', function() {
            const searchQuery = searchInput.value.trim();
            const selectedOptionsString = Array.from(selectedOptions).join(',');
            
            fetch('/simple_search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `query=${encodeURIComponent(searchQuery)}&selected_options=${encodeURIComponent(selectedOptionsString)}`
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    displaySearchResults(data);
                }
            });
        });
    }

    // Handle advanced search
    if (advancedSearchForm) {
        advancedSearchForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = {
                subreddits: formData.get('subreddits').split(',').map(s => s.trim()),
                from_time: formData.get('from_time'),
                to_time: formData.get('to_time'),
                sort_types: Array.from(formData.getAll('sort_types')),
                post_limit: formData.get('post_limit'),
                include_comments: formData.get('include_comments') === 'on',
                search_text: formData.get('search_text'),
                comment_limit: formData.get('comment_limit')
            };
            
            fetch('/advanced_search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    displaySearchResults(data);
                }
            });
        });
    }

    // Display search results
    function displaySearchResults(data) {
        searchResults.classList.remove('hidden');
        document.getElementById('searchId').textContent = data.search_id;
        document.getElementById('resultsCount').textContent = data.results_count;
    }

    // Handle copy search ID
    if (copySearchIdBtn) {
        copySearchIdBtn.addEventListener('click', function() {
            const searchId = document.getElementById('searchId').textContent;
            navigator.clipboard.writeText(searchId).then(() => {
                this.textContent = 'Copied!';
                setTimeout(() => {
                    this.textContent = 'Copy Search ID';
                }, 2000);
            });
        });
    }
});