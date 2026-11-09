/**
 * main.js — Multi-City Real Estate ML Application
 * Handles city persistence (localStorage), dynamic location fetching,
 * client-side form validation, and UI enhancements.
 */

'use strict';

const CITY_STORAGE_KEY = 'estateiq_selected_city';

document.addEventListener('DOMContentLoaded', () => {

    // ── 1. City Preference Management (localStorage sync) ─────────────────────
    const currentCity = document.body.dataset.city || 'Bengaluru';
    const storedCity = localStorage.getItem(CITY_STORAGE_KEY);

    // If local storage has a preferred city and user visits root / with no city query param,
    // we can sync if needed; here we ensure localStorage has the active server city.
    if (!storedCity || storedCity !== currentCity) {
        localStorage.setItem(CITY_STORAGE_KEY, currentCity);
    }

    // Attach click listeners to all city switcher dropdown items
    document.querySelectorAll('.city-dropdown-item, .city-select-trigger').forEach((item) => {
        item.addEventListener('click', (e) => {
            const chosenCity = item.dataset.city;
            if (chosenCity) {
                localStorage.setItem(CITY_STORAGE_KEY, chosenCity);
            }
        });
    });

    // ── 2. Dynamic Location Dropdown on City Change (in forms) ────────────────
    const citySelectorForm = document.getElementById('form_city_select');
    const locationSelect = document.getElementById('location');

    if (citySelectorForm && locationSelect) {
        citySelectorForm.addEventListener('change', async (e) => {
            const selectedCity = e.target.value;
            localStorage.setItem(CITY_STORAGE_KEY, selectedCity);

            locationSelect.disabled = true;
            locationSelect.innerHTML = '<option value="" disabled selected>Loading localities…</option>';

            try {
                const res = await fetch(`/api/locations?city=${encodeURIComponent(selectedCity)}`);
                const data = await res.json();

                locationSelect.innerHTML = `<option value="" disabled selected>Select ${selectedCity} locality…</option>`;
                if (data.status === 'success' && data.locations && data.locations.length > 0) {
                    data.locations.forEach((loc) => {
                        const opt = document.createElement('option');
                        opt.value = loc;
                        opt.textContent = loc;
                        locationSelect.appendChild(opt);
                    });
                } else {
                    locationSelect.innerHTML = '<option value="" disabled selected>No locations found</option>';
                }
            } catch (err) {
                console.error('Failed to load locations for city:', err);
                locationSelect.innerHTML = '<option value="" disabled selected>Error loading locations</option>';
            } finally {
                locationSelect.disabled = false;
            }
        });
    }

    // ── 3. Predict Form Client Validation & Loading Spinner ──────────────────
    const predictForm = document.getElementById('predictForm');
    if (predictForm) {
        predictForm.addEventListener('submit', (e) => {
            const sqftInput = document.getElementById('total_sqft');
            const locSelect = document.getElementById('location');

            if (!sqftInput || parseFloat(sqftInput.value) <= 0) {
                alert('Please enter a valid built-up area (sqft) greater than zero.');
                e.preventDefault();
                return;
            }
            if (locSelect && !locSelect.value) {
                alert('Please select a locality from the dropdown.');
                e.preventDefault();
                return;
            }

            // Show a loading spinner on the submit button
            const btn = predictForm.querySelector('button[type="submit"]');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML =
                    '<span class="spinner-border spinner-border-sm me-2" role="status"></span>' +
                    'Estimating Market Value…';
            }
        });
    }

    // ── 4. Flash-message Auto-Dismiss ────────────────────────────────────────
    document.querySelectorAll('.alert-dismissible').forEach((alert) => {
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            if (bsAlert) bsAlert.close();
        }, 6000);
    });

});
