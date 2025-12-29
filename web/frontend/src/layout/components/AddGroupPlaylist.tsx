import { Button } from "@/components/ui/button";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
	DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useMusicStore } from "@/stores/useMusicStore";
import { Plus, Upload } from "lucide-react";
import { useRef, useState } from "react";
import toast from "react-hot-toast";
import { useChatStore } from "@/stores/useChatStore";
import { useUser } from "@clerk/clerk-react";
import { useEffect } from "react";

interface NewGroupPlaylist {
  name: string;
  members: string[]; 
  user: string;
}

const AddGroupPlaylistDialog = () => {
	const [songDialogOpen, setSongDialogOpen] = useState(false);
	const [isLoading, setIsLoading] = useState(false);
	const { users, fetchUsers, createGroupPlaylist } = useChatStore();
	const { user } = useUser();

	const [newGroup, setNewGroup] = useState<NewGroupPlaylist>({
	  	name: "",
	  	members: [],
		user: "",
	});
	
	useEffect(() => {
		if (!user) return;

		fetchUsers();
		setNewGroup((prev) => ({ ...prev, user: user.id }));
		console.log("User:",user.id);
	}, [fetchUsers, user]);

	const toggleMember = (userId: string) => {
	  setNewGroup((prev) => ({
	    ...prev,
	    members: prev.members.includes(userId)
	      ? prev.members.filter((id) => id !== userId)
	      : [...prev.members, userId],
	  }));
	};

	const handleSubmit = async () => {
	  	setIsLoading(true);

	  	try {
	  	  	if (!newGroup.name.trim()) {
    		  	setIsLoading(false);
    		  	return toast.error("Please provide playlist name");
    		}
		
    		if (newGroup.members.length === 0) {
    		  	setIsLoading(false);
    		  	return toast.error("Please select at least one friend");
    		}
		
    		await createGroupPlaylist(newGroup);
		  
	  	  	toast.success("Group playlist created successfully");
		  
	  	  	setNewGroup({
	  	  	  	name: "",
	  	  	  	members: [],
				user: "",
	  	  	});
		  
	  	  	setSongDialogOpen(false);
		
	  	} catch (error: any) {
	  	  	toast.error("Failed to create playlist: " + error.message);
	  	} finally {
	  	  	setIsLoading(false);
	  	}
	};

	return (
		<Dialog open={songDialogOpen} onOpenChange={setSongDialogOpen}>
			<DialogTrigger asChild>
				<Button className='bg-emerald-500 hover:bg-emerald-600 text-black px-4 py-2 relative z-50'>
					<Plus className='h-4 w-4' />
				</Button>
			</DialogTrigger>

			<DialogContent className='bg-zinc-900 border-zinc-700 max-h-[80vh] overflow-auto'>
				<DialogHeader>
					<DialogTitle>Create Group Playlist</DialogTitle>
					<DialogDescription>Create a shared playlist for your group</DialogDescription>
				</DialogHeader>

				<div className='space-y-4 py-4'>
					<div className='space-y-2'>
					  	<label className='text-sm font-medium'>Playlist Name</label>
					  	<Input
					  	  	value={newGroup.name}
					  	  	onChange={(e) => setNewGroup({ ...newGroup, name: e.target.value })}
					  	  	className='bg-zinc-800 border-zinc-700'
					  	/>
					</div>

					<div className='space-y-2'>
					  	<label className="text-sm font-medium">Add Friends</label>

  						<div className="max-h-48 overflow-y-auto space-y-2 border border-zinc-700 rounded-md p-2">
  						  	{users?.length === 0 && (
  						  	  	<p className="text-sm text-zinc-500">No friends found</p>
  						  	)}

  						  	{users?.map((u) => (
  						  	  	<div
  						  	  	  	key={u._id}
  						  	  	  	className="flex items-center justify-between p-2 hover:bg-zinc-800 rounded-md"
  						  	  	>
  						  	  	  	<div className="flex items-center gap-3">
  						  	  	  	  	<Avatar className='size-10 border border-zinc-800'>
											<AvatarImage src={u.imageUrl} alt={u.fullName} />
											<AvatarFallback>{u.fullName[0]}</AvatarFallback>
										</Avatar>
										<div className='flex-1 min-w-0'>
											<div className='flex items-center gap-2'>
												<span className='font-medium text-sm text-white'>{u.fullName}</span>
											</div>
										</div>
  						  	  	  	</div>

  						  	  	  	<input
  						  	  	  	  	type="checkbox"
  						  	  	  	  	checked={newGroup.members.includes(u._id)}
  						  	  	  	  	onChange={() => toggleMember(u._id)}
  						  	  	  	  	className="w-4 h-4 accent-emerald-500 cursor-pointer"
  						  	  	  	/>
  						  	  	</div>
  						  	))}
  						</div>
					</div>
				</div>

				<DialogFooter>
					<Button variant='outline' onClick={() => setSongDialogOpen(false)} disabled={isLoading}>
						Cancel
					</Button>
					<Button onClick={handleSubmit} disabled={isLoading}>
					  {isLoading ? "Creating..." : "Create Group Playlist"}
					</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
};
export default AddGroupPlaylistDialog;
