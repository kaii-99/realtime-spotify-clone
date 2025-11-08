import axios from "axios";
import { Song } from "../models/song.model.js";

export const getRecommendations = async (req, res) => {
  try {
    const { songIds } = req.query;

    if (!songIds) {
      return res.status(400).json({ message: "songIds query parameter is required" });
    }

    const AI_API_URL = process.env.AI_MODEL_URL || "http://localhost:4000";

    // 🔹 Call Python model API
    const aiResponse = await axios.get(`${AI_API_URL}/recommendations`, {
      params: { songIds },
    });

    // AI model returns an array of recommended song IDs
    const recommendedIds = aiResponse.data;

    // Fetch full song objects from DB
    const recommendedSongs = await Song.find({ _id: { $in: recommendedIds } });

    res.json(recommendedSongs);
  } catch (error) {
    console.error("Error in getRecommendations:", error);
    res.status(500).json({ message: "Internal server error" });
  }
};