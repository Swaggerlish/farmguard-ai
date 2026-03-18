const CACHE_NAME = 'farmguard-v1';
const STATIC_CACHE = 'farmguard-static-v1';
const API_CACHE = 'farmguard-api-v1';

// Assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/src/main.jsx',
  '/src/App.jsx',
  '/src/index.css',
  '/src/components/UploadForm.jsx',
  '/src/components/PredictionCard.jsx',
  '/src/components/LanguageSelector.jsx',
  '/src/components/HistoryView.jsx'
];

// API endpoints to cache
const API_ENDPOINTS = [
  '/api/health'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('Service Worker: Installing...');
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('Service Worker: Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .catch((error) => {
        console.log('Service Worker: Failed to cache static assets', error);
      })
  );
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== STATIC_CACHE && cacheName !== API_CACHE) {
            console.log('Service Worker: Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Handle API requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      caches.open(API_CACHE).then((cache) => {
        return fetch(request)
          .then((response) => {
            // Cache successful API responses
            if (response.status === 200) {
              cache.put(request, response.clone());
            }
            return response;
          })
          .catch(() => {
            // Return cached response if available
            return cache.match(request).then((cachedResponse) => {
              if (cachedResponse) {
                return cachedResponse;
              }
              // Return offline response for predict endpoint
              if (url.pathname === '/api/predict') {
                return new Response(
                  JSON.stringify({
                    error: 'offline',
                    message: 'You are currently offline. Disease detection requires internet connection. Your scan will be saved and synced when connection is restored.',
                    predictions: [],
                    advice: {
                      description: 'Offline mode - please check your internet connection',
                      treatment: 'Connect to internet to get treatment recommendations',
                      prevention: 'Ensure stable internet for best results',
                      urgency: 'unknown'
                    }
                  }),
                  {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' }
                  }
                );
              }
              return new Response('Offline - No cached data available', { status: 503 });
            });
          });
      })
    );
    return;
  }

  // Handle static assets
  event.respondWith(
    caches.match(request)
      .then((response) => {
        if (response) {
          return response;
        }
        return fetch(request);
      })
      .catch(() => {
        // Return offline fallback for navigation requests
        if (request.mode === 'navigate') {
          return caches.match('/index.html');
        }
        return new Response('Offline - Content not available', { status: 503 });
      })
  );
});

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    event.waitUntil(doBackgroundSync());
  }
});

async function doBackgroundSync() {
  console.log('Service Worker: Performing background sync');
  // Handle any queued offline actions here
  // For now, just log that sync occurred
}