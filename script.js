// Sample product data
const products = [
    {
        id: 1,
        title: "Premium Wireless Headphones",
        price: 299.99,
        originalPrice: 399.99,
        image: "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400&h=300&fit=crop",
        rating: 4.5,
        reviews: 2847,
        category: "electronics",
        description: "High-quality wireless headphones with noise cancellation and premium sound quality. Perfect for music lovers and professionals."
    },
    {
        id: 2,
        title: "Smart Fitness Watch",
        price: 199.99,
        originalPrice: 249.99,
        image: "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400&h=300&fit=crop",
        rating: 4.3,
        reviews: 1523,
        category: "electronics",
        description: "Advanced fitness tracking with heart rate monitoring, GPS, and smartphone connectivity. Track your health and fitness goals."
    },
    {
        id: 3,
        title: "Bestselling Mystery Novel",
        price: 14.99,
        originalPrice: 24.99,
        image: "https://images.unsplash.com/photo-1543002588-bfa74002ed7e?w=400&h=300&fit=crop",
        rating: 4.7,
        reviews: 3241,
        category: "books",
        description: "A thrilling mystery that keeps you guessing until the very end. Perfect for fans of suspense and crime fiction."
    },
    {
        id: 4,
        title: "Premium Cotton T-Shirt",
        price: 29.99,
        originalPrice: 49.99,
        image: "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=300&fit=crop",
        rating: 4.4,
        reviews: 856,
        category: "clothing",
        description: "Ultra-soft premium cotton t-shirt with perfect fit and lasting comfort. Available in multiple colors and sizes."
    },
    {
        id: 5,
        title: "Modern Coffee Maker",
        price: 89.99,
        originalPrice: 129.99,
        image: "https://images.unsplash.com/photo-1495474472287-4d71bcdd2085?w=400&h=300&fit=crop",
        rating: 4.2,
        reviews: 1247,
        category: "home",
        description: "Programmable coffee maker with multiple brewing options. Start your day with the perfect cup of coffee every time."
    },
    {
        id: 6,
        title: "Yoga Mat Pro",
        price: 39.99,
        originalPrice: 59.99,
        image: "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=400&h=300&fit=crop",
        rating: 4.6,
        reviews: 2156,
        category: "sports",
        description: "Professional-grade yoga mat with superior grip and cushioning. Perfect for all types of yoga and exercise routines."
    },
    {
        id: 7,
        title: "Building Blocks Set",
        price: 49.99,
        originalPrice: 79.99,
        image: "https://images.unsplash.com/photo-1558060370-d644479cb6f7?w=400&h=300&fit=crop",
        rating: 4.8,
        reviews: 1834,
        category: "toys",
        description: "Creative building blocks set that sparks imagination. Perfect for children aged 6+ to develop problem-solving skills."
    },
    {
        id: 8,
        title: "Wireless Gaming Mouse",
        price: 79.99,
        originalPrice: 99.99,
        image: "https://images.unsplash.com/photo-1527814050087-3793815479db?w=400&h=300&fit=crop",
        rating: 4.4,
        reviews: 967,
        category: "electronics",
        description: "High-precision wireless gaming mouse with customizable buttons and RGB lighting. Dominate your gaming sessions."
    },
    {
        id: 9,
        title: "Self-Help Success Guide",
        price: 19.99,
        originalPrice: 29.99,
        image: "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop",
        rating: 4.5,
        reviews: 2341,
        category: "books",
        description: "Transform your life with proven strategies for success and personal development. A must-read for ambitious individuals."
    },
    {
        id: 10,
        title: "Designer Jeans",
        price: 89.99,
        originalPrice: 149.99,
        image: "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400&h=300&fit=crop",
        rating: 4.3,
        reviews: 1456,
        category: "clothing",
        description: "Premium designer jeans with perfect fit and style. Made from high-quality denim for comfort and durability."
    },
    {
        id: 11,
        title: "Indoor Plant Collection",
        price: 34.99,
        originalPrice: 54.99,
        image: "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=400&h=300&fit=crop",
        rating: 4.7,
        reviews: 1823,
        category: "home",
        description: "Beautiful collection of low-maintenance indoor plants. Perfect for beginners and adds life to any space."
    },
    {
        id: 12,
        title: "Professional Tennis Racket",
        price: 129.99,
        originalPrice: 199.99,
        image: "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?w=400&h=300&fit=crop",
        rating: 4.6,
        reviews: 743,
        category: "sports",
        description: "Professional-grade tennis racket used by pros. Lightweight design with excellent power and control for serious players."
    }
];

// Shopping cart
let cart = [];
let currentFilter = 'all';
let currentSort = 'featured';

// Initialize the website
document.addEventListener('DOMContentLoaded', function() {
    loadProducts();
    setupEventListeners();
    updateCartCount();
});

// Setup event listeners
function setupEventListeners() {
    // Search functionality
    document.getElementById('searchInput').addEventListener('input', handleSearch);
    
    // Category filter in search dropdown
    document.querySelector('.search-category').addEventListener('change', function() {
        currentFilter = this.value;
        loadProducts();
    });
    
    // Close modals when clicking outside
    window.addEventListener('click', function(event) {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            if (event.target === modal) {
                modal.classList.remove('show');
            }
        });
    });
}

// Load and display products
function loadProducts() {
    const grid = document.getElementById('productsGrid');
    let filteredProducts = products;
    
    // Apply category filter
    if (currentFilter !== 'all') {
        filteredProducts = products.filter(product => product.category === currentFilter);
    }
    
    // Apply sorting
    filteredProducts = sortProducts(filteredProducts);
    
    // Generate HTML
    grid.innerHTML = filteredProducts.map(product => `
        <div class="product-card fade-in" onclick="showProductDetails(${product.id})">
            <img src="${product.image}" alt="${product.title}" class="product-image">
            <div class="product-title">${product.title}</div>
            <div class="product-rating">
                <span class="stars">${generateStars(product.rating)}</span>
                <span class="rating-count">(${product.reviews.toLocaleString()})</span>
            </div>
            <div class="product-price">
                <span class="current-price">$${product.price}</span>
                <span class="original-price">$${product.originalPrice}</span>
            </div>
            <div class="free-price">FREE at checkout!</div>
            <button class="add-to-cart" onclick="event.stopPropagation(); addToCart(${product.id})">
                Add to Cart
            </button>
        </div>
    `).join('');
}

// Generate star rating display
function generateStars(rating) {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    let stars = '';
    
    for (let i = 0; i < fullStars; i++) {
        stars += '★';
    }
    
    if (hasHalfStar) {
        stars += '☆';
    }
    
    const remainingStars = 5 - fullStars - (hasHalfStar ? 1 : 0);
    for (let i = 0; i < remainingStars; i++) {
        stars += '☆';
    }
    
    return stars;
}

// Sort products
function sortProducts(products = []) {
    const sortBy = document.getElementById('sortBy').value;
    
    switch (sortBy) {
        case 'price-low':
            return [...products].sort((a, b) => a.price - b.price);
        case 'price-high':
            return [...products].sort((a, b) => b.price - a.price);
        case 'rating':
            return [...products].sort((a, b) => b.rating - a.rating);
        default:
            return products;
    }
}

// Handle sorting change
function sortProducts() {
    loadProducts();
}

// Filter products by category
function filterProducts(category) {
    currentFilter = category;
    document.querySelector('.search-category').value = category;
    loadProducts();
}

// Handle search
function handleSearch() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const grid = document.getElementById('productsGrid');
    
    let filteredProducts = products.filter(product => 
        product.title.toLowerCase().includes(searchTerm) ||
        product.description.toLowerCase().includes(searchTerm)
    );
    
    // Apply category filter
    if (currentFilter !== 'all') {
        filteredProducts = filteredProducts.filter(product => product.category === currentFilter);
    }
    
    // Apply sorting
    filteredProducts = sortProducts(filteredProducts);
    
    // Display results
    if (filteredProducts.length === 0) {
        grid.innerHTML = `
            <div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: #666;">
                <h3>No products found</h3>
                <p>Try adjusting your search terms or browse our categories.</p>
            </div>
        `;
    } else {
        grid.innerHTML = filteredProducts.map(product => `
            <div class="product-card fade-in" onclick="showProductDetails(${product.id})">
                <img src="${product.image}" alt="${product.title}" class="product-image">
                <div class="product-title">${product.title}</div>
                <div class="product-rating">
                    <span class="stars">${generateStars(product.rating)}</span>
                    <span class="rating-count">(${product.reviews.toLocaleString()})</span>
                </div>
                <div class="product-price">
                    <span class="current-price">$${product.price}</span>
                    <span class="original-price">$${product.originalPrice}</span>
                </div>
                <div class="free-price">FREE at checkout!</div>
                <button class="add-to-cart" onclick="event.stopPropagation(); addToCart(${product.id})">
                    Add to Cart
                </button>
            </div>
        `).join('');
    }
}

// Show product details modal
function showProductDetails(productId) {
    const product = products.find(p => p.id === productId);
    if (!product) return;
    
    const modal = document.getElementById('productModal');
    const modalBody = document.getElementById('modalBody');
    
    modalBody.innerHTML = `
        <img src="${product.image}" alt="${product.title}" class="modal-image">
        <div class="modal-info">
            <h2>${product.title}</h2>
            <div class="product-rating mb-20">
                <span class="stars">${generateStars(product.rating)}</span>
                <span class="rating-count">(${product.reviews.toLocaleString()} reviews)</span>
            </div>
            <div class="product-price mb-20">
                <span class="current-price">$${product.price}</span>
                <span class="original-price">$${product.originalPrice}</span>
            </div>
            <div class="free-price mb-20">FREE at checkout!</div>
            <div class="modal-description mb-20">
                ${product.description}
            </div>
            <button class="add-to-cart" onclick="addToCart(${product.id}); closeModal();">
                Add to Cart
            </button>
        </div>
    `;
    
    modal.classList.add('show');
}

// Close product modal
function closeModal() {
    document.getElementById('productModal').classList.remove('show');
}

// Add product to cart
function addToCart(productId) {
    const product = products.find(p => p.id === productId);
    if (!product) return;
    
    const existingItem = cart.find(item => item.id === productId);
    
    if (existingItem) {
        existingItem.quantity += 1;
    } else {
        cart.push({
            ...product,
            quantity: 1
        });
    }
    
    updateCartCount();
    updateCartDisplay();
    showCartAddedNotification(product.title);
}

// Show notification when item added to cart
function showCartAddedNotification(productTitle) {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background-color: #007600;
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        z-index: 4000;
        animation: slideIn 0.3s ease;
    `;
    notification.innerHTML = `<strong>Added to cart:</strong> ${productTitle}`;
    
    document.body.appendChild(notification);
    
    // Remove notification after 3 seconds
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// Update cart count
function updateCartCount() {
    const count = cart.reduce((total, item) => total + item.quantity, 0);
    document.getElementById('cartCount').textContent = count;
}

// Toggle cart sidebar
function toggleCart() {
    const sidebar = document.getElementById('cartSidebar');
    sidebar.classList.toggle('open');
    updateCartDisplay();
}

// Update cart display
function updateCartDisplay() {
    const cartItems = document.getElementById('cartItems');
    const cartTotal = document.getElementById('cartTotal');
    
    if (cart.length === 0) {
        cartItems.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #666;">
                <i class="fas fa-shopping-cart" style="font-size: 48px; margin-bottom: 20px;"></i>
                <p>Your cart is empty</p>
                <p>Start shopping to add items!</p>
            </div>
        `;
        cartTotal.textContent = '$0.00';
        return;
    }
    
    cartItems.innerHTML = cart.map(item => `
        <div class="cart-item">
            <img src="${item.image}" alt="${item.title}" class="cart-item-image">
            <div class="cart-item-info">
                <div class="cart-item-title">${item.title}</div>
                <div class="cart-item-price">$${item.price}</div>
                <div class="quantity-controls">
                    <button class="quantity-btn" onclick="updateQuantity(${item.id}, ${item.quantity - 1})">-</button>
                    <span>${item.quantity}</span>
                    <button class="quantity-btn" onclick="updateQuantity(${item.id}, ${item.quantity + 1})">+</button>
                    <button class="quantity-btn" onclick="removeFromCart(${item.id})" style="margin-left: 10px; background-color: #dc3545; color: white;">Remove</button>
                </div>
            </div>
        </div>
    `).join('');
    
    const total = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
    cartTotal.textContent = '$' + total.toFixed(2);
}

// Update item quantity
function updateQuantity(productId, newQuantity) {
    if (newQuantity <= 0) {
        removeFromCart(productId);
        return;
    }
    
    const item = cart.find(item => item.id === productId);
    if (item) {
        item.quantity = newQuantity;
        updateCartCount();
        updateCartDisplay();
    }
}

// Remove item from cart
function removeFromCart(productId) {
    cart = cart.filter(item => item.id !== productId);
    updateCartCount();
    updateCartDisplay();
}

// Proceed to checkout
function checkout() {
    if (cart.length === 0) {
        alert('Your cart is empty!');
        return;
    }
    
    const modal = document.getElementById('checkoutModal');
    const orderSummary = document.getElementById('orderSummary');
    
    const total = cart.reduce((sum, item) => sum + (item.price * item.quantity), 0);
    
    orderSummary.innerHTML = `
        <h3>Order Summary</h3>
        <div style="margin: 20px 0;">
            ${cart.map(item => `
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span>${item.title} x ${item.quantity}</span>
                    <span>$${(item.price * item.quantity).toFixed(2)}</span>
                </div>
            `).join('')}
        </div>
        <div style="border-top: 1px solid #ddd; padding-top: 15px; display: flex; justify-content: space-between; font-weight: bold;">
            <span>Subtotal:</span>
            <span>$${total.toFixed(2)}</span>
        </div>
        <div style="display: flex; justify-content: space-between; color: #007600; font-weight: bold; font-size: 18px;">
            <span>Your Total:</span>
            <span class="free-price">$0.00 (FREE!)</span>
        </div>
    `;
    
    modal.classList.add('show');
    toggleCart(); // Close cart sidebar
}

// Close checkout modal
function closeCheckoutModal() {
    document.getElementById('checkoutModal').classList.remove('show');
}

// Place order
function placeOrder() {
    const orderNumber = 'FZ-' + Math.random().toString(36).substr(2, 9).toUpperCase();
    
    // Clear cart
    cart = [];
    updateCartCount();
    
    // Close checkout modal
    closeCheckoutModal();
    
    // Show success modal
    document.getElementById('orderNumber').textContent = orderNumber;
    document.getElementById('successModal').classList.add('show');
}

// Close success modal
function closeSuccessModal() {
    document.getElementById('successModal').classList.remove('show');
}

// Scroll to products section
function scrollToProducts() {
    document.querySelector('.main-content').scrollIntoView({ 
        behavior: 'smooth' 
    });
}

// Add CSS for notification animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);