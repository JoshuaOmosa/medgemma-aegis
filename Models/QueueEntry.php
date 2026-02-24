<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class QueueEntry extends Model
{
    use HasFactory;

    protected $fillable = [
        'name',
        'email',
        'service',
        'status',         // if needed
        'queue_number',   // ✅ this line is essential
    ];
}
