# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: replacement.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import firasim_client.libs.common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='replacement.proto',
  package='fira_message.sim_to_ref',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x11replacement.proto\x12\x17\x66ira_message.sim_to_ref\x1a\x0c\x63ommon.proto\"]\n\x10RobotReplacement\x12%\n\x08position\x18\x01 \x01(\x0b\x32\x13.fira_message.Robot\x12\x12\n\nyellowteam\x18\x05 \x01(\x08\x12\x0e\n\x06turnon\x18\x06 \x01(\x08\"?\n\x0f\x42\x61llReplacement\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\n\n\x02vx\x18\x03 \x01(\x01\x12\n\n\x02vy\x18\x04 \x01(\x01\"\x80\x01\n\x0bReplacement\x12\x36\n\x04\x62\x61ll\x18\x01 \x01(\x0b\x32(.fira_message.sim_to_ref.BallReplacement\x12\x39\n\x06robots\x18\x02 \x03(\x0b\x32).fira_message.sim_to_ref.RobotReplacementb\x06proto3')
  ,
  dependencies=[common__pb2.DESCRIPTOR,])




_ROBOTREPLACEMENT = _descriptor.Descriptor(
  name='RobotReplacement',
  full_name='fira_message.sim_to_ref.RobotReplacement',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='position', full_name='fira_message.sim_to_ref.RobotReplacement.position', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='yellowteam', full_name='fira_message.sim_to_ref.RobotReplacement.yellowteam', index=1,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='turnon', full_name='fira_message.sim_to_ref.RobotReplacement.turnon', index=2,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=60,
  serialized_end=153,
)


_BALLREPLACEMENT = _descriptor.Descriptor(
  name='BallReplacement',
  full_name='fira_message.sim_to_ref.BallReplacement',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='fira_message.sim_to_ref.BallReplacement.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='fira_message.sim_to_ref.BallReplacement.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vx', full_name='fira_message.sim_to_ref.BallReplacement.vx', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='vy', full_name='fira_message.sim_to_ref.BallReplacement.vy', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=155,
  serialized_end=218,
)


_REPLACEMENT = _descriptor.Descriptor(
  name='Replacement',
  full_name='fira_message.sim_to_ref.Replacement',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ball', full_name='fira_message.sim_to_ref.Replacement.ball', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='robots', full_name='fira_message.sim_to_ref.Replacement.robots', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=221,
  serialized_end=349,
)

_ROBOTREPLACEMENT.fields_by_name['position'].message_type = common__pb2._ROBOT
_REPLACEMENT.fields_by_name['ball'].message_type = _BALLREPLACEMENT
_REPLACEMENT.fields_by_name['robots'].message_type = _ROBOTREPLACEMENT
DESCRIPTOR.message_types_by_name['RobotReplacement'] = _ROBOTREPLACEMENT
DESCRIPTOR.message_types_by_name['BallReplacement'] = _BALLREPLACEMENT
DESCRIPTOR.message_types_by_name['Replacement'] = _REPLACEMENT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

RobotReplacement = _reflection.GeneratedProtocolMessageType('RobotReplacement', (_message.Message,), dict(
  DESCRIPTOR = _ROBOTREPLACEMENT,
  __module__ = 'replacement_pb2'
  # @@protoc_insertion_point(class_scope:fira_message.sim_to_ref.RobotReplacement)
  ))
_sym_db.RegisterMessage(RobotReplacement)

BallReplacement = _reflection.GeneratedProtocolMessageType('BallReplacement', (_message.Message,), dict(
  DESCRIPTOR = _BALLREPLACEMENT,
  __module__ = 'replacement_pb2'
  # @@protoc_insertion_point(class_scope:fira_message.sim_to_ref.BallReplacement)
  ))
_sym_db.RegisterMessage(BallReplacement)

Replacement = _reflection.GeneratedProtocolMessageType('Replacement', (_message.Message,), dict(
  DESCRIPTOR = _REPLACEMENT,
  __module__ = 'replacement_pb2'
  # @@protoc_insertion_point(class_scope:fira_message.sim_to_ref.Replacement)
  ))
_sym_db.RegisterMessage(Replacement)


# @@protoc_insertion_point(module_scope)
