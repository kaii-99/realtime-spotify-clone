import { ListeningHistory } from "../models/listeningHistory.model.js";
import { User } from "../models/user.model.js";
import axios from "axios";

export const recordListenHistory = async (req, res, next) => {
	try {
		const { songId, clerkId } = req.body;

		if (!clerkId || !songId) {
			return res.status(400).json({ error: "Missing clerkId or songId" });
		}

		const user = await User.findOne({ clerkId });
		if (!user) {
			return res.status(404).json({ error: "User not found" });
		}

		const AI_API_URL = process.env.AI_MODEL_URL || "http://localhost:4000";

    	// 🔹 Call Python model API
    	const aiResponse = await axios.get(`${AI_API_URL}/weather-mood`, {
    	  params: { 
    	    city: req.query.city || "Singapore"
    	  },
    	});

    	// API returns current weather
    	const weatherData = aiResponse.data;

        console.log("Recording history: ", user._id, songId, weatherData)

		await ListeningHistory.create({
			userId: user._id,
			songId,
			timestamp: new Date(),
			weather: weatherData.weather,
  			timeOfDay: weatherData.time_of_day
		});

		res.status(200).json({ message: "Listening history recorded" });
	} catch (error) {
		next(error);
	}
};

// Get all songs a user has listened to
export const getUserListeningHistory = async (req, res, next) => {
  try {
    const clerkId = req.params.userId;

    if (!clerkId) {
      return res.status(400).json({ error: "Missing clerkId" });
    }

    const user = await User.findOne({ clerkId });
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    // Fetch all listening history for this user
    const history = await ListeningHistory.find({ userId: user._id }).lean();

    res.status(200).json(history); // or res.json(songs) if populated
  } catch (error) {
    next(error);
  }
};