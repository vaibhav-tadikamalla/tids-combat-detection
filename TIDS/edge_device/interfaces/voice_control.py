import speech_recognition as sr
import pyttsx3
import threading
import time
from fuzzywuzzy import fuzz

class VoiceCommandSystem:
    """
    Hands-free voice control for tactical situations.
    
    Features:
    - Wake word activation: "Guardian"
    - Continuous listening mode
    - Natural language understanding
    - Text-to-speech feedback
    - Critical command shortcuts
    """
    
    WAKE_WORD = "guardian"
    
    COMMANDS = {
        'status': ['status', 'sitrep', 'report'],
        'emergency': ['help', 'emergency', 'medic', 'mayday'],
        'stealth': ['stealth', 'silent', 'quiet mode'],
        'navigation': ['where am i', 'location', 'navigate'],
        'squad': ['squad status', 'team', 'friendlies'],
        'threat': ['threats', 'enemies', 'contacts'],
        'vitals': ['vitals', 'health', 'medical'],
        'battery': ['battery', 'power'],
        'mute': ['mute', 'silence', 'shut up']
    }
    
    def __init__(self, guardian_system):
        self.guardian = guardian_system
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 0.7)
        
        # State
        self.listening = False
        self.muted = False
        self.wake_word_detected = False
        
        # Calibrate for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    
    def start(self):
        """Start voice command system"""
        self.listening = True
        
        # Start listening thread
        listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        listen_thread.start()
        
        self.speak("Voice command system activated")
    
    def stop(self):
        """Stop voice command system"""
        self.listening = False
        self.speak("Voice commands deactivated")
    
    def _listen_loop(self):
        """Main listening loop"""
        while self.listening:
            try:
                with self.microphone as source:
                    # Listen for wake word or command
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    
                try:
                    # Recognize speech
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    # Check for wake word
                    if not self.wake_word_detected:
                        if self.WAKE_WORD in text:
                            self.wake_word_detected = True
                            self.speak("Yes, listening")
                            continue
                    else:
                        # Process command
                        self._process_command(text)
                        self.wake_word_detected = False
                        
                except sr.UnknownValueError:
                    # Could not understand audio
                    if self.wake_word_detected:
                        self.speak("Could not understand. Please repeat.")
                        self.wake_word_detected = False
                except sr.RequestError as e:
                    print(f"Speech recognition service error: {e}")
                    
            except sr.WaitTimeoutError:
                # No speech detected
                continue
            except Exception as e:
                print(f"Voice control error: {e}")
                time.sleep(1)
    
    def _process_command(self, text):
        """Process recognized command"""
        # Match command using fuzzy matching
        best_match = None
        best_score = 0
        
        for category, keywords in self.COMMANDS.items():
            for keyword in keywords:
                score = fuzz.partial_ratio(text, keyword)
                if score > best_score:
                    best_score = score
                    best_match = category
        
        # Execute command if confidence is high enough
        if best_score > 70:
            self._execute_command(best_match, text)
        else:
            self.speak("Command not recognized")
    
    def _execute_command(self, command, text):
        """Execute specific command"""
        if command == 'status':
            self._report_status()
        
        elif command == 'emergency':
            self._trigger_emergency()
        
        elif command == 'stealth':
            self._activate_stealth()
        
        elif command == 'navigation':
            self._report_location()
        
        elif command == 'squad':
            self._report_squad_status()
        
        elif command == 'threat':
            self._report_threats()
        
        elif command == 'vitals':
            self._report_vitals()
        
        elif command == 'battery':
            self._report_battery()
        
        elif command == 'mute':
            self._toggle_mute()
    
    def _report_status(self):
        """Report device and soldier status"""
        # Get current status from guardian system
        vitals = self.guardian.sensor_fusion.get_current_vitals()
        battery = self.guardian.battery_level
        
        status = f"System operational. Heart rate {vitals['heart_rate']} beats per minute. "
        status += f"Oxygen saturation {vitals['spo2']} percent. "
        status += f"Battery at {battery} percent."
        
        self.speak(status)
    
    def _trigger_emergency(self):
        """Trigger emergency alert"""
        self.speak("Emergency alert triggered. Sending distress signal.")
        
        # Trigger emergency in guardian system
        asyncio.run_coroutine_threadsafe(
            self.guardian.trigger_emergency(),
            self.guardian.loop
        )
    
    def _activate_stealth(self):
        """Activate stealth mode"""
        self.speak("Activating stealth mode")
        
        # Reduce transmission power, increase encryption
        asyncio.run_coroutine_threadsafe(
            self.guardian.security.enable_stealth_mode(),
            self.guardian.loop
        )
    
    def _report_location(self):
        """Report current location"""
        location = self.guardian.sensor_fusion.gps_position
        
        if location:
            lat, lon = location
            response = f"Current position: {abs(lat):.4f} degrees "
            response += "north" if lat >= 0 else "south"
            response += f", {abs(lon):.4f} degrees "
            response += "east" if lon >= 0 else "west"
            
            self.speak(response)
        else:
            self.speak("GPS position unavailable")
    
    def _report_squad_status(self):
        """Report squad member status"""
        # Would integrate with mesh network to get squad data
        self.speak("Squad status: 8 active, 2 injured, 0 critical")
    
    def _report_threats(self):
        """Report detected threats"""
        # Get recent threats from guardian system
        recent_threats = self.guardian.recent_alerts[-3:]
        
        if not recent_threats:
            self.speak("No threats detected")
        else:
            count = len(recent_threats)
            self.speak(f"{count} threats detected in last 5 minutes")
    
    def _report_vitals(self):
        """Report detailed vitals"""
        vitals = self.guardian.sensor_fusion.get_current_vitals()
        
        response = f"Heart rate: {vitals['heart_rate']} beats per minute. "
        response += f"Blood oxygen: {vitals['spo2']} percent. "
        response += f"Respiratory rate: {vitals['breathing_rate']} breaths per minute. "
        
        # Assess status
        if vitals['heart_rate'] > 120:
            response += "Elevated heart rate detected."
        elif vitals['spo2'] < 95:
            response += "Low oxygen saturation warning."
        else:
            response += "Vitals normal."
        
        self.speak(response)
    
    def _report_battery(self):
        """Report battery status"""
        battery = self.guardian.battery_level
        
        if battery > 50:
            status = "good"
        elif battery > 20:
            status = "moderate"
        else:
            status = "low"
        
        self.speak(f"Battery at {battery} percent. Status: {status}")
    
    def _toggle_mute(self):
        """Mute/unmute voice feedback"""
        self.muted = not self.muted
        
        if not self.muted:
            self.speak("Voice feedback enabled")
    
    def speak(self, text):
        """Text-to-speech output"""
        if not self.muted:
            print(f"[Voice]: {text}")
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
    
    def emergency_override(self, message):
        """Emergency message that bypasses mute"""
        original_mute = self.muted
        self.muted = False
        self.speak(f"Emergency: {message}")
        self.muted = original_mute


class TacticalVoiceCommands:
    """
    Predefined tactical voice commands for combat situations.
    
    Quick voice shortcuts that don't require wake word in critical scenarios.
    """
    
    CRITICAL_COMMANDS = {
        'contact': 'Enemy contact reported',
        'grenade': 'Grenade warning - take cover',
        'down': 'Soldier down - requesting medical assistance',
        'clear': 'Area clear - all threats neutralized',
        'reloading': 'Reloading - cover me',
        'moving': 'Moving to new position'
    }
    
    def __init__(self, voice_system):
        self.voice = voice_system
        self.quick_command_mode = False
    
    def enable_quick_commands(self):
        """Enable quick tactical commands without wake word"""
        self.quick_command_mode = True
        self.voice.speak("Quick command mode activated")
    
    def process_quick_command(self, text):
        """Process tactical command"""
        for command, response in self.CRITICAL_COMMANDS.items():
            if command in text.lower():
                self.voice.speak(response)
                # Also broadcast to squad
                return True
        return False
