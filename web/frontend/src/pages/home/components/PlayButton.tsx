import { Button } from "@/components/ui/button";
import { usePlayerStore } from "@/stores/usePlayerStore";
import { Song } from "@/types";
import { Pause, Play } from "lucide-react";
import { useUser } from "@clerk/clerk-react"; 
import { axiosInstance } from "@/lib/axios";

const recordListening = async (songId: string, clerkId: string) => {
	try {
		await axiosInstance.post("/listen-history", {
			songId,
			clerkId, // or omit if using auth in backend
		});
	} catch (error: any) {
		console.error("Failed to record listening:", error.message);
	}
};

const PlayButton = ({ song }: { song: Song }) => {
	const { currentSong, isPlaying, setCurrentSong, togglePlay } = usePlayerStore();
	const isCurrentSong = currentSong?._id === song._id;
	const { user } = useUser();

	const handlePlay = async() => {
		if (isCurrentSong) {
			togglePlay();
		} else {
			setCurrentSong(song);

			if (user?.id) {
				recordListening(song._id, user.id);
			}
		}
	};

	return (
		<Button
			size={"icon"}
			onClick={handlePlay}
			className={`absolute bottom-3 right-2 bg-green-500 hover:bg-green-400 hover:scale-105 transition-all 
				opacity-0 translate-y-2 group-hover:translate-y-0 ${
					isCurrentSong ? "opacity-100" : "opacity-0 group-hover:opacity-100"
				}`}
		>
			{isCurrentSong && isPlaying ? (
				<Pause className='size-5 text-black' />
			) : (
				<Play className='size-5 text-black' />
			)}
		</Button>
	);
};
export default PlayButton;
