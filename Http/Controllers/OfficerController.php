<?php

namespace App\Http\Controllers;

use App\Models\QueueEntry;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Mail;

class OfficerController extends Controller
{
    /**
     * Show the officer dashboard with all service queues.
     */
    public function dashboard()
    {
        $services = config('services_list.services');
        $queues = [];

        foreach ($services as $service) {
            $queues[$service] = QueueEntry::where('service', $service)
                ->whereIn('status', ['waiting', 'in_service'])
                ->get();
        }

        return view('officer.dashboard', compact('queues'));
    }

    /**
     * Move the next person in queue to 'in_service' and notify them.
     */
   public function callNext(Request $request)
{
    $service = $request->input('service');

    $next = \App\Models\QueueEntry::where('service', $service)
        ->where('status', 'waiting')
        ->orderBy('created_at')
        ->first();

    if ($next) {
        $next->status = 'in_service';
        $next->save();

        // Only send if we have a valid queue entry
        \Mail::raw("Hello {$next->name}, it's your turn for {$next->service}.", function ($message) use ($next) {
            $message->to($next->email)
                    ->subject('Now Serving');
        });
    } else {
        // Optional: Add flash message or silent fallback
        session()->flash('error', 'No one is currently waiting in this queue.');
    }

    return redirect()->back();
}


    /**
     * Mark the current person as 'completed'.
     */
    public function complete($id)
    {
        $entry = QueueEntry::findOrFail($id);
        $entry->status = 'completed';
        $entry->save();

        return redirect()->back();
    }
}
