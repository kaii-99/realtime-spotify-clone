import { ListeningHistory } from "../models/listeningHistory.model.js";
import { User } from "../models/user.model.js";

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

        console.log("Recording history: ", user._id, songId)

		await ListeningHistory.create({
			userId: user._id,
			songId,
			timestamp: new Date(),
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