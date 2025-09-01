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
