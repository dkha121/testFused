# User - System action
LIST_USER_ACT = "user_action: inform_intent, negate_intent, affirm_intent, inform, request, affirm, negate, select," \
                "request_alts, thank, bye, greet, general"
LIST_SYSTEM_ACT_KETOD = "inform, request, confirm, offer, notify_success, notify_failure, inform_count, offer_intent," \
                        "req_more, bye, greet, welcome, general"
LIST_SYSTEM_ACT_FUSED = "inform, request, select, recommend, no_offer, book, no_book, offer_book, offer_booked," \
                        "req_more, bye, greet, welcome, general"

# Schema-guided
Rental_Car = "Rental_Car; Slots: type, car_name, pickup_location, pickup_date, pickup_time, city, end_date, " \
             "total_price, price_per_day, add_insurance"
RideSharing = "RideSharing; Slots: destination, shared_ride, ride_fare, approximate_ride_duration, number_of_riders, " \
              "ride_type, wait_time, number_of_seats"
Buses = "Buses; Slots: origin, destination, from_station, to_station, departure_date, departure_time, price, " \
        "num_passengers, transfers, fare_type, additional_luggage, category"

Flight = "Flight; Slots: origin_city, passengers, seating_class, destination_city, origin_airpot, " \
         "destination_airpot, depature_date, return_date, number_stops, outbound_departure_time, " \
         "outbound_arrival_time, inbound_arrival_time, inbound_departure_time, price, refundable, airlines, is_nonstop"
Train = "Train; Slots: from, destination, departure, day, leaveat, price, bookpeople, class, trip_protection, arrive, id, " \
        "choice, name, ref, duration"
Taxi = "Taxi; Slots: arriveby, type, departure, destination, leaveat, phone, choice, bookday, bookpeople"
Message = "Message; Slots: location, contact_name"

Hotel = "Hotel; Slots: address, number_of_rooms, check_in_date, number_of_days, rating, hotel_name, address, " \
        "phone_number, price_per_night, price, has_wifi, number_of_adults, check_out_date, pets_welcome, " \
        "smoky_allows, has_laundry_service, bookpeople, choice, bookstay, parking, stars, postcode, area, type, ref"
Restaurants = "Restaurants; Slots: restaurant_name, date, time, serves_alcohol, has_live_music, phone_number, " \
              "address, number_of_seats, price_range, city, cuisine, has_seating_outdoors, has_vegetarian_options, " \
              "rating, postcode, ref, choice, area"
Travel = "Travel; Slots: location, attraction_name, category, phone_number, free_entry, good_for_kids"
Booking = "Booking: Slot: bookday, bookpeople, bookstay, booktime, name, ref"
WEATHER = "Weather; Slots: precipitation, humidity, wind, temperature, city, date"
HOSPITAL = "Hospital; Slots: address, phone, postcode, department, name"
COSMETIC = "Services; Slots: stylist_name, phone_number, average_rating, is_unisex, street_address, city, type, " \
             "appointment_date, appointment_time, denstist_name, offer_cosmetic_services, doctor_name, therapist_name"
HOME = "Home; Slots: area, address, property_name, phone_number, furnished, pets_allowed, intent, visit_date," \
       "numer_of_beds, number_of_baths, has-garage, in_unit_laundry, price"
Media = "Media; Slots: title, genre, subtile, director, actors, price"
Music = "Music; Slots: song_name, artist, album, genre, year, device"
Movies = "Movies; Slots: price, number_of_tickets, show_type, theater_name, show_time, show_date, genre, " \
         "street_address, location, movie_name, aggregate_rating, starring, director"
Events = "Events; Slots: category, subcategory, event_name, date, time, number_of_seats, city_of_event, " \
         "event_location, address_of_location, event_type, number_of_tickets, venue, venue_addressm, price_per_ticket"
CALENDAR = "Calendar; Slots: event_date, event_time, event_location, event_name, available_start_time, available_end_time"
ATTRACTION = "Attraction; Slots: address, area, name, choice, entrancefee, openhours, phone, postcode, parking, type"
POLICE = "Police; Slot: address, phone, postcode, name, department"